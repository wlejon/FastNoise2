#include <cassert>
#include <cmath>
#include <cstring>
#include <utility>

#include "Generator.h"

#pragma warning( disable:4250 )

using namespace FastNoise;

static constexpr size_t kRegisterSize = std::max<size_t>( 4, FS::NativeRegisterCount<float>() * 2 );
using float32v = FS::Register<float, kRegisterSize>;
using int32v = FS::Register<std::int32_t, kRegisterSize>;
using mask32v = typename float32v::MaskType;

// A coordinate of the positions a generator is called with: an origin in
// double, the same for every lane, and each lane's offset from it in float.
// A float holds a position millions of units from zero (a planet's surface)
// only to a fraction of a unit, too coarse for a lattice noise's cells and
// the fractions it interpolates over; with the origin in double, a lattice
// noise rebases on its own cell of the origin (Rebase) and works in small
// numbers however far out the positions are. Scaling a position by a
// constant scales both parts; offsets computed per lane add to the lane's
// part. With an origin of 0 every generator computes exactly what it did
// with a plain float.
struct PosV
{
    float32v v;
    double o = 0.0;

    PosV() = default;
    FS_FORCEINLINE PosV( float32v lanes ) : v( lanes ) {}
    FS_FORCEINLINE PosV( float32v lanes, double origin ) : v( lanes ), o( origin ) {}

    FS_FORCEINLINE PosV& operator*=( float s ) { v *= float32v( s ); o *= (double)s; return *this; }
    FS_FORCEINLINE PosV& operator+=( float32v d ) { v += d; return *this; }
};

FS_FORCEINLINE static PosV operator*( PosV p, float s ) { return p *= s; }
FS_FORCEINLINE static PosV operator+( PosV p, float32v d ) { return p += d; }
FS_FORCEINLINE static PosV operator-( PosV p, float32v d ) { p.v -= d; return p; }
FS_FORCEINLINE static PosV operator+( PosV a, PosV b ) { return { a.v + b.v, a.o + b.o }; }
FS_FORCEINLINE static PosV operator-( PosV a, PosV b ) { return { a.v - b.v, a.o - b.o }; }

// a * k + b and b - a * k, as FS::FMulAdd and FS::FNMulAdd.
FS_FORCEINLINE static PosV PosFMulAdd( PosV a, float k, PosV b ) { return { FS::FMulAdd( a.v, float32v( k ), b.v ), a.o * (double)k + b.o }; }
FS_FORCEINLINE static PosV PosFNMulAdd( PosV a, float k, PosV b ) { return { FS::FNMulAdd( a.v, float32v( k ), b.v ), b.o - a.o * (double)k }; }

// The position as one float (exact only near zero), for a generator whose
// value is a function of the position itself rather than of a lattice.
FS_FORCEINLINE static float32v PosFlat( PosV p )
{
    return p.o == 0.0 ? p.v : float32v( (float)p.o ) + p.v;
}

// The lattice cell holding the origin (an integer, taken mod 2^32 as the
// hashes take it) and each lane's position from that cell's corner: a
// lattice noise floors, rounds and hashes those, adding the cell to its own.
FS_FORCEINLINE static float32v Rebase( PosV p, int32_t& cell )
{
    if( p.o == 0.0 )
    {
        cell = 0;
        return p.v;
    }
    double c = std::floor( p.o );
    cell = (int32_t)(uint32_t)(int64_t)c;
    return p.v + float32v( (float)( p.o - c ) );
}

// The position less a whole number of `period`s of its origin, for a
// generator periodic in it.
FS_FORCEINLINE static float32v PosWrap( PosV p, double period )
{
    return p.o == 0.0 ? p.v : p.v + float32v( (float)( p.o - period * std::floor( p.o / period ) ) );
}

// The position rounded to an integer as FS::Convert rounds (the rebased lanes
// rounded, plus the origin's cell, itself an integer).
FS_FORCEINLINE static int32v PosRound( PosV p )
{
    int32_t cell;
    float32v lanes = Rebase( p, cell );
    return FS::Convert<int32_t>( lanes ) + int32v( cell );
}

// The same for a lattice noise working in skewed coordinates (the simplex
// family): each axis' origin plus `skew` times the sum of them all, split
// into its cell and the fraction the lanes' skewed positions add.
template<size_t N>
FS_FORCEINLINE static void RebaseSkewed( const PosV ( &p )[N], float skew, int32_t ( &cell )[N], float ( &fraction )[N] )
{
    double sum = 0.0;
    for( size_t i = 0; i < N; i++ ) sum += p[i].o;
    for( size_t i = 0; i < N; i++ )
    {
        double s = p[i].o + (double)skew * sum;
        double c = std::floor( s );
        cell[i] = (int32_t)(uint32_t)(int64_t)c;
        fraction[i] = (float)( s - c );
    }
}

// An integer times a prime, wrapping as the hashes' arithmetic does.
FS_FORCEINLINE static int32_t PrimedCell( int32_t cell, int32_t prime )
{
    return (int32_t)( (uint32_t)cell * (uint32_t)prime );
}

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<Generator, SIMD> : public virtual Generator
{
public:
    virtual float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y ) const = 0;
    virtual float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y, PosV z ) const = 0;
    virtual float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y, PosV z, PosV w ) const { return Gen( seed, x, y, z ); }

#define FASTNOISE_IMPL_GEN_T\
    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y ) const override { return GenT( seed, x, y ); }\
    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y, PosV z ) const override { return GenT( seed, x, y, z ); }\
    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y, PosV z, PosV w ) const override { return GenT( seed, x, y, z, w ); }

    FastSIMD::FeatureSet GetActiveFeatureSet() const final
    {
        return FastSIMD::FeatureSetDefault();
    }

    using VoidPtrStorageType = const DispatchClass<Generator, SIMD>*;

    void SetSourceSIMDPtr( const Generator* base, const void** simdPtr ) final
    {
        if( !base )
        {
            *simdPtr = nullptr;
            return;
        }
        auto simd = dynamic_cast<VoidPtrStorageType>( base );

        assert( simd );
        *simdPtr = reinterpret_cast<const void*>( simd );
    }

    template<typename T, typename... POS>
    static FS_FORCEINLINE float32v FS_VECTORCALL GetSourceValue( const FastNoise::HybridSourceT<T>& memberVariable, int32v seed, POS... pos )
    {
        if( memberVariable.simdGeneratorPtr )
        {
            auto simdGen = reinterpret_cast<VoidPtrStorageType>( memberVariable.simdGeneratorPtr );

            return simdGen->Gen( seed, pos... );
        }
        return float32v( memberVariable.constant );
    }

    template<typename T, typename... POS>
    static FS_FORCEINLINE float32v FS_VECTORCALL GetSourceValue( const FastNoise::GeneratorSourceT<T>& memberVariable, int32v seed, POS... pos )
    {
        assert( memberVariable.simdGeneratorPtr );
        auto simdGen = reinterpret_cast<VoidPtrStorageType>( memberVariable.simdGeneratorPtr );

        return simdGen->Gen( seed, pos... );
    }

    template<typename T>
    static FS_FORCEINLINE const DispatchClass<T, SIMD>* GetSourceSIMD( const FastNoise::GeneratorSourceT<T>& memberVariable )
    {
        assert( memberVariable.simdGeneratorPtr );
        auto simdGen = reinterpret_cast<VoidPtrStorageType>( memberVariable.simdGeneratorPtr );

        auto simdT = static_cast<const FastSIMD::DispatchClass<T, SIMD>*>( simdGen );
        return simdT;
    }

    FastNoise::OutputMinMax GenUniformGrid2D( float* noiseOut, double xOffset, double yOffset, int xCount, int yCount, float xStepSize, float yStepSize, int seed ) const final
    {
        float32v min( kInfinity );
        float32v max( -kInfinity );

        int32v xIdx( 0 );
        int32v yIdx( 0 );
        float32v xScale( xStepSize );
        float32v yScale( yStepSize );

        int32v xCountV( xCount );
        int32v xMax = xCountV + int32v( -1 );

        intptr_t totalValues = xCount * yCount;
        intptr_t index = 0;

        xIdx += FS::LoadIncremented<int32v>();

        AxisReset<true>( xIdx, yIdx, xMax, xCountV, xCount );

        while( index < totalValues - (intptr_t)int32v::ElementCount )
        {
            PosV xPos( FS::Convert<float>( xIdx ) * xScale, xOffset );
            PosV yPos( FS::Convert<float>( yIdx ) * yScale, yOffset );

            float32v gen = Gen( int32v( seed ), xPos, yPos );
            FS::Store( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS::Min( min, gen );
            max = FS::Max( max, gen );
#endif

            index += int32v::ElementCount;
            xIdx += int32v( int32v::ElementCount );

            AxisReset<false>( xIdx, yIdx, xMax, xCountV, xCount );
        }

        PosV xPos( FS::Convert<float>( xIdx ) * xScale, xOffset );
        PosV yPos( FS::Convert<float>( yIdx ) * yScale, yOffset );

        float32v gen = Gen( int32v( seed ), xPos, yPos );

        return StoreRemaining( noiseOut, totalValues, index, min, max, gen );
    }

    FastNoise::OutputMinMax GenUniformGrid3D( float* noiseOut, double xOffset, double yOffset, double zOffset, int xCount, int yCount, int zCount, float xStepSize, float yStepSize, float zStepSize, int seed ) const final
    {
        float32v min( kInfinity );
        float32v max( -kInfinity );

        int32v xIdx( 0 );
        int32v yIdx( 0 );
        int32v zIdx( 0 );
        float32v xScale( xStepSize );
        float32v yScale( yStepSize );
        float32v zScale( zStepSize );

        int32v xCountV( xCount );
        int32v xMax = xCountV + int32v( -1 );
        int32v yCountV( yCount );
        int32v yMax = yCountV + int32v( -1 );

        intptr_t totalValues = xCount * yCount * zCount;
        intptr_t index = 0;

        xIdx += FS::LoadIncremented<int32v>();

        AxisReset<true>( xIdx, yIdx, xMax, xCountV, xCount );
        AxisReset<true>( yIdx, zIdx, yMax, yCountV, xCount * yCount );

        while( index < totalValues - (intptr_t)int32v::ElementCount )
        {
            PosV xPos( FS::Convert<float>( xIdx ) * xScale, xOffset );
            PosV yPos( FS::Convert<float>( yIdx ) * yScale, yOffset );
            PosV zPos( FS::Convert<float>( zIdx ) * zScale, zOffset );

            float32v gen = Gen( int32v( seed ), xPos, yPos, zPos );
            FS::Store( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS::Min( min, gen );
            max = FS::Max( max, gen );
#endif

            index += int32v::ElementCount;
            xIdx += int32v( int32v::ElementCount );

            AxisReset<false>( xIdx, yIdx, xMax, xCountV, xCount );
            AxisReset<false>( yIdx, zIdx, yMax, yCountV, xCount * yCount );
        }

        PosV xPos( FS::Convert<float>( xIdx ) * xScale, xOffset );
        PosV yPos( FS::Convert<float>( yIdx ) * yScale, yOffset );
        PosV zPos( FS::Convert<float>( zIdx ) * zScale, zOffset );

        float32v gen = Gen( int32v( seed ), xPos, yPos, zPos );

        return StoreRemaining( noiseOut, totalValues, index, min, max, gen );
    }

    FastNoise::OutputMinMax GenUniformGrid4D( float* noiseOut, double xOffset, double yOffset, double zOffset, double wOffset, int xCount, int yCount, int zCount, int wCount, float xStepSize, float yStepSize, float zStepSize, float wStepSize, int seed ) const final
    {
        float32v min( kInfinity );
        float32v max( -kInfinity );

        int32v xIdx( 0 );
        int32v yIdx( 0 );
        int32v zIdx( 0 );
        int32v wIdx( 0 );
        float32v xScale( xStepSize );
        float32v yScale( yStepSize );
        float32v zScale( zStepSize );
        float32v wScale( wStepSize );

        int32v xCountV( xCount );
        int32v xMax = xCountV + int32v( -1 );
        int32v yCountV( yCount );
        int32v yMax = yCountV + int32v( -1 );
        int32v zCountV( zCount );
        int32v zMax = zCountV + int32v( -1 );

        intptr_t totalValues = xCount * yCount * zCount * wCount;
        intptr_t index = 0;

        xIdx += FS::LoadIncremented<int32v>();

        AxisReset<true>( xIdx, yIdx, xMax, xCountV, xCount );
        AxisReset<true>( yIdx, zIdx, yMax, yCountV, xCount * yCount );
        AxisReset<true>( zIdx, wIdx, zMax, zCountV, xCount * yCount * zCount );

        while( index < totalValues - (intptr_t)int32v::ElementCount )
        {
            PosV xPos( FS::Convert<float>( xIdx ) * xScale, xOffset );
            PosV yPos( FS::Convert<float>( yIdx ) * yScale, yOffset );
            PosV zPos( FS::Convert<float>( zIdx ) * zScale, zOffset );
            PosV wPos( FS::Convert<float>( wIdx ) * wScale, wOffset );

            float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );
            FS::Store( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS::Min( min, gen );
            max = FS::Max( max, gen );
#endif

            index += int32v::ElementCount;
            xIdx += int32v( int32v::ElementCount );

            AxisReset<false>( xIdx, yIdx, xMax, xCountV, xCount );
            AxisReset<false>( yIdx, zIdx, yMax, yCountV, xCount * yCount );
            AxisReset<false>( zIdx, wIdx, zMax, zCountV, xCount * yCount * zCount );
        }

        PosV xPos( FS::Convert<float>( xIdx ) * xScale, xOffset );
        PosV yPos( FS::Convert<float>( yIdx ) * yScale, yOffset );
        PosV zPos( FS::Convert<float>( zIdx ) * zScale, zOffset );
        PosV wPos( FS::Convert<float>( wIdx ) * wScale, wOffset );

        float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );

        return StoreRemaining( noiseOut, totalValues, index, min, max, gen );
    }

    FastNoise::OutputMinMax GenPositionArray2D( float* noiseOut, int count, const float* xPosArray, const float* yPosArray, double xOffset, double yOffset, int seed ) const final
    {
        float32v min( kInfinity );
        float32v max( -kInfinity );

        intptr_t index = 0;
        while( index < count - (intptr_t)int32v::ElementCount )
        {
            PosV xPos( FS::Load<float32v>( &xPosArray[index] ), xOffset );
            PosV yPos( FS::Load<float32v>( &yPosArray[index] ), yOffset );

            float32v gen = Gen( int32v( seed ), xPos, yPos );
            FS::Store( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS::Min( min, gen );
            max = FS::Max( max, gen );
#endif
            index += int32v::ElementCount;
        }

        PosV xPos( LoadRemaining( xPosArray, count, index ), xOffset );
        PosV yPos( LoadRemaining( yPosArray, count, index ), yOffset );

        float32v gen = Gen( int32v( seed ), xPos, yPos );

        return StoreRemaining<true>( noiseOut, count, index, min, max, gen );
    }

    FastNoise::OutputMinMax GenPositionArray3D( float* noiseOut, int count, const float* xPosArray, const float* yPosArray, const float* zPosArray, double xOffset, double yOffset, double zOffset, int seed ) const final
    {
        float32v min( kInfinity );
        float32v max( -kInfinity );

        intptr_t index = 0;
        while( index < count - (intptr_t)int32v::ElementCount )
        {
            PosV xPos( FS::Load<float32v>( &xPosArray[index] ), xOffset );
            PosV yPos( FS::Load<float32v>( &yPosArray[index] ), yOffset );
            PosV zPos( FS::Load<float32v>( &zPosArray[index] ), zOffset );

            float32v gen = Gen( int32v( seed ), xPos, yPos, zPos );
            FS::Store( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS::Min( min, gen );
            max = FS::Max( max, gen );
#endif
            index += int32v::ElementCount;
        }

        PosV xPos( LoadRemaining( xPosArray, count, index ), xOffset );
        PosV yPos( LoadRemaining( yPosArray, count, index ), yOffset );
        PosV zPos( LoadRemaining( zPosArray, count, index ), zOffset );

        float32v gen = Gen( int32v( seed ), xPos, yPos, zPos );

        return StoreRemaining<true>( noiseOut, count, index, min, max, gen );
    }

    FastNoise::OutputMinMax GenPositionArray4D( float* noiseOut, int count, const float* xPosArray, const float* yPosArray, const float* zPosArray, const float* wPosArray, double xOffset, double yOffset, double zOffset, double wOffset, int seed ) const final
    {
        float32v min( kInfinity );
        float32v max( -kInfinity );

        intptr_t index = 0;
        while( index < count - (intptr_t)int32v::ElementCount )
        {
            PosV xPos( FS::Load<float32v>( &xPosArray[index] ), xOffset );
            PosV yPos( FS::Load<float32v>( &yPosArray[index] ), yOffset );
            PosV zPos( FS::Load<float32v>( &zPosArray[index] ), zOffset );
            PosV wPos( FS::Load<float32v>( &wPosArray[index] ), wOffset );

            float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );
            FS::Store( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS::Min( min, gen );
            max = FS::Max( max, gen );
#endif
            index += int32v::ElementCount;
        }

        PosV xPos( LoadRemaining( xPosArray, count, index ), xOffset );
        PosV yPos( LoadRemaining( yPosArray, count, index ), yOffset );
        PosV zPos( LoadRemaining( zPosArray, count, index ), zOffset );
        PosV wPos( LoadRemaining( wPosArray, count, index ), wOffset );

        float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );

        return StoreRemaining<true>( noiseOut, count, index, min, max, gen );
    }

    float GenSingle2D( double x, double y, int seed ) const final
    {
        return FS::Extract0( Gen( int32v( seed ), PosV( float32v( 0 ), x ), PosV( float32v( 0 ), y ) ) );
    }

    float GenSingle3D( double x, double y, double z, int seed ) const final
    {
        return FS::Extract0( Gen( int32v( seed ), PosV( float32v( 0 ), x ), PosV( float32v( 0 ), y ), PosV( float32v( 0 ), z ) ) );
    }

    float GenSingle4D( double x, double y, double z, double w, int seed ) const final
    {
        return FS::Extract0( Gen( int32v( seed ), PosV( float32v( 0 ), x ), PosV( float32v( 0 ), y ), PosV( float32v( 0 ), z ), PosV( float32v( 0 ), w ) ) );
    }

    FastNoise::OutputMinMax GenTileable2D( float* noiseOut, int xSize, int ySize, float xStepSize, float yStepSize, int seed ) const final
    {
        float32v min( kInfinity );
        float32v max( -kInfinity );

        int32v xIdx( 0 );
        int32v yIdx( 0 );

        int32v xSizeV( xSize );
        int32v ySizeV( ySize );
        int32v xMax = xSizeV + int32v( -1 );

        intptr_t totalValues = xSize * ySize;
        intptr_t index = 0;

        float pi2Recip( 0.15915493667f );
        float xSizePi = (float)xSize * pi2Recip;
        float ySizePi = (float)ySize * pi2Recip;
        float32v xFreq = float32v( xSizePi * xStepSize );
        float32v yFreq = float32v( ySizePi * yStepSize );
        float32v xMul = float32v( 1 / xSizePi );
        float32v yMul = float32v( 1 / ySizePi );

        xIdx += FS::LoadIncremented<int32v>();

        AxisReset<true>( xIdx, yIdx, xMax, xSizeV, xSize );

        while( index < totalValues - (intptr_t)int32v::ElementCount )
        {
            float32v xF = FS::Convert<float>( xIdx ) * xMul;
            float32v yF = FS::Convert<float>( yIdx ) * yMul;

            float32v xPos = FS::Cos( xF ) * xFreq;
            float32v yPos = FS::Cos( yF ) * yFreq;
            float32v zPos = FS::Sin( xF ) * xFreq;
            float32v wPos = FS::Sin( yF ) * yFreq;

            float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );
            FS::Store( &noiseOut[index], gen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS::Min( min, gen );
            max = FS::Max( max, gen );
#endif

            index += int32v::ElementCount;
            xIdx += int32v( int32v::ElementCount );

            AxisReset<false>( xIdx, yIdx, xMax, xSizeV, xSize );
        }

        float32v xF = FS::Convert<float>( xIdx ) * xMul;
        float32v yF = FS::Convert<float>( yIdx ) * yMul;

        float32v xPos = FS::Cos( xF ) * xFreq;
        float32v yPos = FS::Cos( yF ) * yFreq;
        float32v zPos = FS::Sin( xF ) * xFreq;
        float32v wPos = FS::Sin( yF ) * yFreq;

        float32v gen = Gen( int32v( seed ), xPos, yPos, zPos, wPos );

        return StoreRemaining( noiseOut, totalValues, index, min, max, gen );
    }

private:
    template<bool INITIAL>
    static FS_FORCEINLINE void AxisReset( int32v& aIdx, int32v& bIdx, int32v aMax, int32v aSize, size_t aStep )
    {
        for( size_t resetLoop = INITIAL ? aStep : 0; resetLoop < int32v::ElementCount; resetLoop += aStep )
        {
            mask32v aReset = aIdx > aMax;
            bIdx = FS::MaskedIncrement( aReset, bIdx );
            aIdx = FS::MaskedSub( aReset, aIdx, aSize );
        }
    }

    static FS_FORCEINLINE float32v LoadRemaining( const float* loadPtr, intptr_t totalValues, intptr_t index )        
    {
        if( index == 0 )
        {
            intptr_t remaining = totalValues - index;

            float32v load;
            std::memcpy( &load, loadPtr, remaining * sizeof( float ) );
            return load;
        }

        return FS::Load<float32v>( &loadPtr[totalValues - float32v::ElementCount] );
    }

    template<bool LOADREMAINING = false>
    static FS_FORCEINLINE FastNoise::OutputMinMax StoreRemaining( float* noiseOut, intptr_t totalValues, intptr_t index, float32v min, float32v max, float32v finalGen )
    {
        FastNoise::OutputMinMax minMax;
        intptr_t remaining = totalValues - index;

        if( LOADREMAINING ? index == 0 : remaining != (intptr_t)int32v::ElementCount )
        {
            std::memcpy( &noiseOut[index], &finalGen, remaining * sizeof( float ) );

#if FASTNOISE_CALC_MIN_MAX
            do
            {
                minMax << noiseOut[index];
            }
            while( ++index < totalValues );
#endif
        }
        else
        {
            FS::Store( &noiseOut[totalValues - float32v::ElementCount], finalGen );

#if FASTNOISE_CALC_MIN_MAX
            min = FS::Min( min, finalGen );
            max = FS::Max( max, finalGen );
#endif
        }

#if FASTNOISE_CALC_MIN_MAX
        float* minP = reinterpret_cast<float*>(&min);
        float* maxP = reinterpret_cast<float*>(&max);
        for( size_t i = 0; i < int32v::ElementCount; i++ )
        {
            minMax << FastNoise::OutputMinMax{ minP[i], maxP[i] };
        }
#endif

        return minMax;
    }

    int32_t ReferencesFetchAdd( int32_t add ) const noexcept final
    {
        if( add )
        {
            return mReferences.fetch_add( add, std::memory_order_relaxed );
        }

        return mReferences.load( std::memory_order_relaxed );
    }
    
    mutable std::atomic<uint32_t> mReferences = 0;
};

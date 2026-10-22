#include "Modifiers.h"

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<DomainScale, SIMD> final : public virtual DomainScale, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;
    
    template<typename... P> 
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        return this->GetSourceValue( mSource, seed, (pos * mScale)... );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<DomainOffset, SIMD> final : public virtual DomainOffset, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;
    
    template<typename... P> 
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        return GenT( std::index_sequence_for<P...>(), seed, pos... );
    }

    template<size_t... I, typename... P>
    FS_FORCEINLINE float32v GenT( std::index_sequence<I...>, int32v seed, P... pos ) const
    {
        // Braced init evaluates left to right; expanding into the call is right to left on the MS ABI, which changes fast-math output
        PosV offsetPos[] = { ( pos + this->GetSourceValue( mOffset[I], seed, pos... ) )... };

        return this->GetSourceValue( mSource, seed, offsetPos[I]... );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<DomainRotate, SIMD> final : public virtual DomainRotate, public DispatchClass<Generator, SIMD>
{
    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y ) const
    {
        if( mPitchSin == 0.0f && mRollSin == 0.0f )
        {
            return this->GetSourceValue( mSource, seed,
                PosFNMulAdd( y, mYawSin, x * mYawCos ),
                PosFMulAdd( x, mYawSin, y * mYawCos ) );
        }

        return Gen( seed, x, y, float32v( 0 ) );
    }

    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y, PosV z ) const
    {
        return this->GetSourceValue( mSource, seed,
            PosFMulAdd( x, mXa, PosFMulAdd( y, mXb, z * mXc ) ),
            PosFMulAdd( x, mYa, PosFMulAdd( y, mYb, z * mYc ) ),
            PosFMulAdd( x, mZa, PosFMulAdd( y, mZb, z * mZc ) ) );
    }

    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y, PosV z, PosV w ) const
    {
        // No rotation for 4D yet
        return this->GetSourceValue( mSource, seed, x, y, z, w );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<SeedOffset, SIMD> final : public virtual SeedOffset, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        return this->GetSourceValue( mSource, seed + int32v( mOffset ), pos... );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<Remap, SIMD> final : public virtual Remap, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v source = this->GetSourceValue( mSource, seed, pos... );

        float32v fromMin = this->GetSourceValue( mFromMin, seed, pos... );
        float32v fromMax = this->GetSourceValue( mFromMax, seed, pos... );
        float32v toMin = this->GetSourceValue( mToMin, seed, pos... );
        float32v toMax = this->GetSourceValue( mToMax, seed, pos... );

        float32v result = toMin + ( ( source - fromMin ) / ( fromMax - fromMin ) * ( toMax - toMin ) );

        if( mClampOutput == Boolean::True )
        {
            result = FS::Max( result, toMin );
            result = FS::Min( result, toMax );
        }

        return result;
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<ConvertRGBA8, SIMD> final : public virtual ConvertRGBA8, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v source = this->GetSourceValue( mSource, seed, pos... );
        
        source = FS::Min( source, float32v( mMax ));
        source = FS::Max( source, float32v( mMin ));
        source -= float32v( mMin );

        source *= float32v( 255.0f / (mMax - mMin) );

        int32v byteVal = FS::Convert<std::int32_t>( source );

        int32v output = int32v( 255 << 24 );
        output |= byteVal;
        output |= byteVal << 8;
        output |= byteVal << 16;

        return FS::Cast<float>( output );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<Terrace, SIMD> final : public virtual Terrace, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v source = this->GetSourceValue( mSource, seed, pos... );

        source *= float32v( mStepCount );
        float32v rounded = FS::Round( source );

        if( mSmoothness.simdGeneratorPtr )
        {
            // Dynamic smoothness from generator
            float32v smoothness = this->GetSourceValue( mSmoothness, seed, pos... );
            mask32v smoothnessMask = smoothness != float32v( 0.0f );

            if( FS::AnyMask( smoothnessMask ) )
            {
                float32v diff = rounded - source;
                mask32v diffSign = diff < float32v( 0 );

                diff = FS::Abs( diff );
                diff = float32v( 0.5f ) - diff;

                float32v smoothnessRecip = float32v( 1.0f ) + FS::Reciprocal( smoothness );
                diff *= smoothnessRecip;
                diff = FS::Min( diff, float32v( 0.5f ) );
                diff = FS::Select( diffSign, float32v( 0.5f ) - diff, diff - float32v( 0.5f ) );

                rounded = FS::Select( smoothnessMask, rounded + diff, rounded );
            }
        }
        else if( mSmoothness.constant != 0.0f )
        {
            // Constant smoothness, use precomputed reciprocal
            float32v diff = rounded - source;
            mask32v diffSign = diff < float32v( 0 );

            diff = FS::Abs( diff );
            diff = float32v( 0.5f ) - diff;

            diff *= float32v( mSmoothnessRecip );
            diff = FS::Min( diff, float32v( 0.5f ) );
            diff = FS::Select( diffSign, float32v( 0.5f ) - diff, diff - float32v( 0.5f ) );

            rounded += diff;
        }

        return rounded * float32v( mStepCountRecip );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<PingPong, SIMD> final : public virtual PingPong, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    static float32v PingPong( float32v t )
    {
        t -= FS::Floor( t * float32v( 0.5f ) ) * float32v( 2 );
        return FS::Select( t < float32v( 1 ), t, float32v( 2 ) - t );
    }

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v pingPongStrength = this->GetSourceValue( mPingPongStrength, seed, pos... );
        return PingPong( this->GetSourceValue( mSource, seed, pos... ) * pingPongStrength );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<DomainAxisScale, SIMD> final : public virtual DomainAxisScale, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        size_t idx = 0;
        ((pos *= mScale[idx++]), ...);

        return this->GetSourceValue( mSource, seed, pos... );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<AddDimension, SIMD> final : public virtual AddDimension, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        if constexpr( sizeof...(P) == (size_t)FastNoise::Dim::Count )
        {
            return this->GetSourceValue( mSource, seed, pos... );
        }
        else
        {
            return this->GetSourceValue( mSource, seed, pos..., this->GetSourceValue( mNewDimensionPosition, seed, pos... ) );
        }
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<RemoveDimension, SIMD> final : public virtual RemoveDimension, public DispatchClass<Generator, SIMD>
{
    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y ) const
    {
        return this->GetSourceValue( mSource, seed, x, y );
    }

    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y, PosV z ) const
    {
        switch( mRemoveDimension )
        {
        case FastNoise::Dim::X:
            return this->GetSourceValue( mSource, seed, y, z );
        case FastNoise::Dim::Y:
            return this->GetSourceValue( mSource, seed, x, z );
        case FastNoise::Dim::Z:
            return this->GetSourceValue( mSource, seed, x, y );
        default:
            return this->GetSourceValue( mSource, seed, x, y, z );
        }
    }

    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y, PosV z, PosV w ) const
    {
        switch( mRemoveDimension )
        {
        case FastNoise::Dim::X:
            return this->GetSourceValue( mSource, seed, y, z, w );
        case FastNoise::Dim::Y:
            return this->GetSourceValue( mSource, seed, x, z, w );
        case FastNoise::Dim::Z:
            return this->GetSourceValue( mSource, seed, x, y, w );
        case FastNoise::Dim::W:
            return this->GetSourceValue( mSource, seed, x, y, z );
        default:
            return this->GetSourceValue( mSource, seed, x, y, z, w );
        }
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<GeneratorCache, SIMD> final : public virtual GeneratorCache, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        thread_local static const void* CachedGenerator = nullptr;
        thread_local static std::int32_t CachedSeed[int32v::ElementCount];
        thread_local static float CachedPos[sizeof...(P)][int32v::ElementCount];
        thread_local static double CachedOrigin[sizeof...(P)];
        thread_local static float CachedValue[int32v::ElementCount];
        // TLS is not always aligned (compiler bug), need to avoid using SIMD types

        const PosV arrayPos[] = { pos... };

        bool isSame = (CachedGenerator == mSource.simdGeneratorPtr);
        isSame &= !FS::AnyMask( seed != FS::Load<int32v>( CachedSeed ) );

        for( size_t i = 0; i < sizeof...( P ); i++ )
        {
            isSame &= !FS::AnyMask( arrayPos[i].v != FS::Load<float32v>( CachedPos[i] ) );
            isSame &= arrayPos[i].o == CachedOrigin[i];
        }

        if( !isSame )
        {
            CachedGenerator = mSource.simdGeneratorPtr;

            float32v value = this->GetSourceValue( mSource, seed, pos... );

            FS::Store( CachedValue, value );
            FS::Store( CachedSeed, seed );

            for( size_t i = 0; i < sizeof...(P); i++ )
            {
                FS::Store( CachedPos[i], arrayPos[i].v );
                CachedOrigin[i] = arrayPos[i].o;
            }

            return value;
        }

        return FS::Load<float32v>( CachedValue );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<SignedSquareRoot, SIMD> final : public virtual SignedSquareRoot, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v value = this->GetSourceValue( mSource, seed, pos... );
        
        return FastLengthSqrt( FS::Abs( value ) ) | FS::SignBit( value );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<Abs, SIMD> final : public virtual Abs, public DispatchClass<Generator, SIMD>
{
    FASTNOISE_IMPL_GEN_T;

    template<typename... P>
    FS_FORCEINLINE float32v GenT( int32v seed, P... pos ) const
    {
        float32v value = this->GetSourceValue( mSource, seed, pos... );
        
        return FS::Abs( value );
    }
};

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<DomainRotatePlane, SIMD> final : public virtual DomainRotatePlane, public DispatchClass<Generator, SIMD>
{
    FS_FORCEINLINE void FS_VECTORCALL RotateCoords( PlaneRotationType rotationType, PosV& x, PosV& y, PosV& z ) const
    {
        PosV newX = x;
        PosV newY = y;
        PosV newZ = z;

        switch( rotationType )
        {
        case PlaneRotationType::ImproveXYPlanes:
        {
            PosV xy = x + y;
            PosV s2 = xy * -0.211324865405187f;
            newZ = z * 0.577350269189626f;
            newX = x + s2 - newZ;
            newY = y + s2 - newZ;
            newZ = newZ + xy * 0.577350269189626f;
        }
            break;

        case PlaneRotationType::ImproveXZPlanes:
        {
            PosV xz = x + z;
            PosV s2 = xz * -0.211324865405187f;
            newY = y * 0.577350269189626f;
            newX = x + s2 - newY;
            newZ = z + s2 - newY;
            newY = newY + xz * 0.577350269189626f;

        }
            break;
        }
        x = newX;
        y = newY;
        z = newZ;
    }

    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y ) const
    {
        PosV z = float32v( 0 );
        RotateCoords( PlaneRotationType::ImproveXYPlanes, x, y, z );

        return this->GetSourceValue( mSource, seed, x, y, z );
    }

    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y, PosV z ) const
    {
        RotateCoords( mRotationType, x, y, z );
        return this->GetSourceValue( mSource, seed, x, y, z );
    }

    float32v FS_VECTORCALL Gen( int32v seed, PosV x, PosV y, PosV z, PosV w ) const
    {
        RotateCoords( mRotationType, x, y, z );
        return this->GetSourceValue( mSource, seed, x, y, z, w );
    }
};

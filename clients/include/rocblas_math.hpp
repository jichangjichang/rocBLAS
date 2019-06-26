/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef ROCBLAS_MATH_H_
#define ROCBLAS_MATH_H_

#include <hip/hip_runtime.h>

#include <string>
#include <sstream>

#include <cmath>
#include <immintrin.h>

#include "rocblas.h"

/* ============================================================================================ */
// Helper routine to convert floats into their half equivalent; uses F16C instructions
inline __host__ rocblas_half float_to_half(float val)
{
    return _cvtss_sh(val, 0);
}

// Helper routine to convert halfs into their floats equivalent; uses F16C instructions
inline __host__ float half_to_float(const rocblas_half val)
{
    return _cvtsh_ss(val);
}

/* ============================================================================================ */
// Helper routine to convert rocblas_type into string
template <typename T, typename std::enable_if<!is_complex<T>>::type* = nullptr>
inline std::string rocblas_type_to_string(const T& value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}

template <typename T, typename std::enable_if<is_complex<T>>::type* = nullptr>
inline std::string rocblas_type_to_string(const T& value)
{
    std::stringstream ss;
    if(value.y >= 0)
        ss << value.x << '+' << value.y << 'i';
    else
        ss << value.x << value.y << 'i';
    return ss.str();
}

inline std::string rocblas_type_to_string(const rocblas_half& value)
{
    std::stringstream ss;
    ss << half_to_float(value);
    return ss.str();
}

/* ============================================================================================ */
/*! \brief  returns true if value is NaN */

template <typename T>
inline bool rocblas_isnan(T)
{
    return false;
}
inline bool rocblas_isnan(double arg)
{
    return std::isnan(arg);
}
inline bool rocblas_isnan(float arg)
{
    return std::isnan(arg);
}
inline bool rocblas_isnan(rocblas_half arg)
{
    return (~arg & 0x7c00) == 0 && (arg & 0x3ff) != 0;
}

/* ============================================================================================ */
// Helper routine to convert two double argument to rocblas type
template <typename T, typename std::enable_if<!is_complex<T>>::type* = nullptr>
inline T compose_value(double real, double imag)
{
    return rocblas_isnan(real) ? 0 : real;
}

template <typename T, typename std::enable_if<is_complex<T>>::type* = nullptr>
inline T compose_value(double real, double imag)
{
    if (rocblas_isnan(real) || rocblas_isnan(imag))
        return T{0.0 , 0.0};
    else
        return T{real, imag};
}

template <>
inline rocblas_half compose_value(double real, double imag)
{
    return rocblas_isnan(real) ? 0 : float_to_half(real);
}

/* ============================================================================================ */
/*! \brief negate a value */

template <class T>
inline T negate(T x)
{
    return -x;
}

template <>
inline rocblas_half negate(rocblas_half x)
{
    return x ^ 0x8000;
}

template <>
inline rocblas_bfloat16 negate(rocblas_bfloat16 x)
{
    x.data = x.data ^ 0x8000;
    return x;
}
#endif

/*******************************************************************************

   Functors for binary arithmetic operators.

   This file is part of XAD, a comprehensive C++ library for
   automatic differentiation.

   Copyright (C) 2010-2024 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#pragma once
#include <XAD/Macros.hpp>
#include <immintrin.h> // For SIMD intrinsics
#include <type_traits> // For std::is_same
namespace xad
{
template <class Scalar>
struct add_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const {
        // Optimize for double using SIMD
        if (std::is_same<Scalar, double>::value) {
            //__m128d vec_a = _mm_set_sd(a); // Load a into a SIMD register
            //__m128d vec_b = _mm_set_sd(b); // Load b into a SIMD register
            return _mm_cvtsd_f64(_mm_add_sd(_mm_set_sd(a), _mm_set_sd(b))); // Perform SIMD addition
            //return _mm_cvtsd_f64(result); // Extract the result
        }
        // Optimize for float using SIMD
        else if (std::is_same<Scalar, float>::value) {
            //__m128 vec_a = _mm_set_ps(0.0f, a, 0.0f, 0.0f); // Load a
            //__m128 vec_b = _mm_set_ps(0.0f, b, 0.0f, 0.0f); // Load b
            //__m128 result = _mm_add_ps(vec_a, vec_b); // Perform SIMD addition
            //return _mm_cvtss_f32(result); // Extract the result
            return _mm_cvtss_f32(_mm_add_ps(_mm_set_ps(0.0f, a, 0.0f, 0.0f), _mm_set_ps(0.0f, b, 0.0f, 0.0f))); // Perform SIMD addition
        }
        else {
            return a + b; // Regular addition
        }
    }
    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }

    XAD_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(1); }
};

template <class Scalar>
struct prod_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a * b; }
     //XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { 
	     //if constexpr (std::is_same<Scalar, double>::value) {
		     //__m128d vec_a = _mm_set_sd(a);
		     //__m128d vec_b = _mm_set_sd(b);
		     //__m128d result = _mm_mul_sd(vec_a, vec_b);
		     //return _mm_cvtsd_f64(result);
	     //} else if constexpr (std::is_same<Scalar, float>::value) {
		     //__m128 vec_a = _mm_set_ps(0.0f, a, 0.0f, 0.0f);
		     //__m128 vec_b = _mm_set_ps(0.0f, b, 0.0f, 0.0f);
		     //__m128 result = _mm_mul_ps(vec_a, vec_b);
		     //return _mm_cvtss_f32(result);
	     //} else {
		     //return a * b; // Fallback for other types
	     //}
     //}

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar& b) const { return b; }

    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar&) const { return a; }
};

template <class Scalar>
struct sub_op
{
    XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a - b; }
	 //XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const {
		 //if constexpr (std::is_same<Scalar, double>::value) {
			 //__m128d vec_a = _mm_set_sd(a);
			 //__m128d vec_b = _mm_set_sd(b);
			 //__m128d result = _mm_sub_sd(vec_a, vec_b);
			 //return _mm_cvtsd_f64(result);
		 //} else if constexpr (std::is_same<Scalar, float>::value) {
			 //__m128 vec_a = _mm_set_ps(0.0f, a, 0.0f, 0.0f);
			 //__m128 vec_b = _mm_set_ps(0.0f, b, 0.0f, 0.0f);
			 //__m128 result = _mm_sub_ps(vec_a, vec_b);
			 //return _mm_cvtss_f32(result);
		 //} else {
			 //return a - b; // Fallback for other types
		 //}
	 //}
    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar&) const { return Scalar(1); }

    XAD_INLINE Scalar derivative_b(const Scalar&, const Scalar&) const { return Scalar(-1); }
};

template <class Scalar>
struct div_op
{
   XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const { return a / b; }
	 //XAD_INLINE Scalar operator()(const Scalar& a, const Scalar& b) const {
		     //if constexpr (std::is_same<Scalar, double>::value) {
			     //__m128d vec_a = _mm_set_sd(a);
			     //__m128d vec_b = _mm_set_sd(b);
			     //// if (b == 0) throw std::invalid_argument("Division by zero.");
			     //__m128d result = _mm_div_sd(vec_a, vec_b); // Note: direct division not available for double
			     //return _mm_cvtsd_f64(result);
		     //} else if constexpr (std::is_same<Scalar, float>::value) {
			     //__m128 vec_a = _mm_set_ps(0.0f, a, 0.0f, 0.0f);
			     //__m128 vec_b = _mm_set_ps(0.0f, b, 0.0f, 0.0f);
			     ////if (b == 0) throw std::invalid_argument("Division by zero.");
			     //__m128 result = _mm_div_ps(vec_a, vec_b);
			     //return _mm_cvtss_f32(result);
		     //} else {
			     //return a / b; // Fallback for other types
		     //}
	 //}

    XAD_INLINE Scalar derivative_a(const Scalar&, const Scalar& b) const { return Scalar(1) / b; }

    XAD_INLINE Scalar derivative_b(const Scalar& a, const Scalar& b) const { return -a / (b * b); }
};
}  // namespace xad

#pragma once

#include <array>
#include <cstdint>
#include <cassert>
#include <cstddef>
#include <ostream>
#include <span>
#include <source_location>
#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/preprocessor.hpp>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SOA_HOST_ONLY __host__
#define SOA_DEVICE_ONLY __device__
#define SOA_HOST_DEVICE __host__ __device__
#define SOA_INLINE __forceinline__
#else
#define SOA_HOST_ONLY
#define SOA_DEVICE_ONLY
#define SOA_HOST_DEVICE
#define SOA_INLINE inline __attribute__((always_inline))
#endif

// Exception throwing (or willful crash in kernels)
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)

#define SOA_THROW_OUT_OF_RANGE(A, I, R)                                      \
  {                                                                          \
    if constexpr (decltype(I)::mode == ffx::soa::range_checking::extended) { \
      printf("%s: index %llu out of range %llu at file %s line %d\n",        \
             (A),                                                            \
             static_cast<unsigned long long>(I.value_),                      \
             static_cast<unsigned long long>(R),                             \
             (I.location_.file_name()),                                      \
             static_cast<int>(I.location_.line()));                          \
    } else {                                                                 \
      printf("%s: index %llu out of range %llu\n",                           \
             (A),                                                            \
             static_cast<unsigned long long>(I.value_),                      \
             static_cast<unsigned long long>(R));                            \
    }                                                                        \
    __trap();                                                                \
  }

#elif defined(__HIPCC__) && defined(__HIP_DEVICE_COMPILE__)

#define SOA_THROW_OUT_OF_RANGE(A, I, R)                                      \
  {                                                                          \
    if constexpr (decltype(I)::mode == ffx::soa::range_checking::extended) { \
      printf("%s: index %llu out of range %llu at file %s line %d\n",        \
             (A),                                                            \
             static_cast<unsigned long long>(I.value_),                      \
             static_cast<unsigned long long>(R),                             \
             (I.location_.file_name()),                                      \
             static_cast<int>(I.location_.line()));                          \
    } else {                                                                 \
      printf("%s: index %llu out of range %llu\n",                           \
             (A),                                                            \
             static_cast<unsigned long long>(I.value_),                      \
             static_cast<unsigned long long>(R));                            \
    }                                                                        \
    abort();                                                                 \
  }

#else
#define SOA_THROW_OUT_OF_RANGE(A, I, R) \
  { ffx::soa::detail::throwOutOfRangeError<decltype(I)::mode>((A), (I), (R)); }
#endif

// declare "scalars" (one value shared across the whole SoA) and "columns" (one value per element)
#define _VALUE_TYPE_SCALAR 0
#define _VALUE_TYPE_COLUMN 1
#define _VALUE_TYPE_EIGEN_COLUMN 2
#define _VALUE_TYPE_METHOD 3
#define _VALUE_TYPE_CONST_METHOD 4
#define _VALUE_TYPE_BLOCK 5
#define _VALUE_TYPE_VIEW_METHOD 6
#define _VALUE_TYPE_CONST_VIEW_METHOD 7
// declare the value of last valid column
#define _VALUE_LAST_COLUMN_TYPE _VALUE_TYPE_EIGEN_COLUMN
// declare a macro useful for passing a valid but not used value
#define _VALUE_TYPE_UNUSED BOOST_PP_LIMIT_MAG

namespace ffx::soa {

  using size_type = std::size_t;
  using byte_size_type = std::size_t;

  enum class SoAColumnType {
    scalar = _VALUE_TYPE_SCALAR,
    column = _VALUE_TYPE_COLUMN,
    eigen = _VALUE_TYPE_EIGEN_COLUMN
  };

  namespace restrict_qualify {

    constexpr bool enabled = true;
    constexpr bool disabled = false;

    constexpr bool default_value = enabled;

  }  // namespace restrict_qualify

  namespace range_checking {

    enum class Mode { Disabled, Enabled, Extended };

    constexpr auto enabled = Mode::Enabled;
    constexpr auto disabled = Mode::Disabled;
    constexpr auto extended = Mode::Extended;

    constexpr auto default_value = enabled;

  }  // namespace range_checking

  template <typename T, bool TRestrictQualify>
  struct add_restrict {};

  template <typename T>
  struct add_restrict<T, restrict_qualify::enabled> {
    using value = T;
    using pointer = T* __restrict__;
    using reference = T& __restrict__;
    using const_value = const T;
    using pointer_to_const = const T* __restrict__;
    using reference_to_const = const T& __restrict__;
  };

  template <typename T>
  struct add_restrict<T, restrict_qualify::disabled> {
    using value = T;
    using pointer = T*;
    using reference = T&;
    using const_value = const T;
    using pointer_to_const = const T*;
    using reference_to_const = const T&;
  };

  // Matryoshka template to avoid commas inside macros
  template <std::size_t TAlignment>
  struct LayoutParameters {
    template <bool TAlignmentEnforcement>
    struct AlignmentEnforcement {
      template <template <std::size_t, bool> typename T>
      using Layout = T<TAlignment, TAlignmentEnforcement>;
    };
  };

  // forward declarations
  template <SoAColumnType TColumnType, typename T>
  struct SoAConstParametersImpl;

  template <SoAColumnType TColumnType, typename T>
  struct SoAParametersImpl;

  // templated const parameter sets for scalars, columns and eigen columns
  template <SoAColumnType TColumnType, typename T>
  struct SoAConstParametersImpl {
    static constexpr SoAColumnType columnType = TColumnType;

    using ValueType = T;
    using ScalarType = T;

    // default constructor
    SoAConstParametersImpl() = default;

    // constructor from address and size
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAConstParametersImpl(ScalarType const* addr) : addr_(addr) {}

    // constructor from a non-const parameter set
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAConstParametersImpl(SoAParametersImpl<columnType, ValueType> const& o)
        : addr_{o.addr_} {}

    SOA_HOST_DEVICE SOA_INLINE ScalarType const* data() const { return addr_; }

    // scalar or column
    ScalarType const* addr_ = nullptr;
  };

  // templated const parameter specialization for Eigen columns
  template <typename T>
  struct SoAConstParametersImpl<SoAColumnType::eigen, T> {
    static constexpr auto columnType = SoAColumnType::eigen;

    using ValueType = T;
    using ScalarType = typename T::Scalar;

    // default constructor
    SoAConstParametersImpl() = default;

    // constructor from individual address, stride and size
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAConstParametersImpl(ScalarType const* addr, byte_size_type stride)
        : addr_(addr), stride_(stride) {}

    // constructor from a non-const parameter set
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAConstParametersImpl(SoAParametersImpl<columnType, ValueType> const& o)
        : addr_{o.addr_}, stride_{o.stride_} {}

    SOA_HOST_DEVICE SOA_INLINE ScalarType const* data() const { return addr_; }
    SOA_HOST_DEVICE SOA_INLINE byte_size_type stride() const { return stride_; }

    // address, stride and size
    ScalarType const* addr_ = nullptr;
    byte_size_type stride_ = 0;
  };

  // Matryoshka template to avoid commas inside macros
  template <SoAColumnType TColumnType>
  struct SoAConstParameters_ColumnType {
    template <typename T>
    using DataType = SoAConstParametersImpl<TColumnType, T>;
  };

  // Templated parameter sets for scalars, columns and Eigen columns
  template <SoAColumnType TColumnType, typename T>
  struct SoAParametersImpl {
    static constexpr SoAColumnType columnType = TColumnType;

    using ValueType = T;
    using ScalarType = T;

    using ConstType = SoAConstParametersImpl<columnType, ValueType>;
    friend ConstType;

    // default constructor
    SoAParametersImpl() = default;

    // constructor from address and size
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl(ScalarType* addr) : addr_(addr) {}

    SOA_HOST_DEVICE SOA_INLINE ScalarType* data() const { return addr_; }

    // scalar or column
    ScalarType* addr_ = nullptr;
  };

  // Templated parameter specialization for Eigen columns
  template <typename T>
  struct SoAParametersImpl<SoAColumnType::eigen, T> {
    static constexpr auto columnType = SoAColumnType::eigen;

    using ValueType = T;
    using ScalarType = T::Scalar;

    using ConstType = SoAConstParametersImpl<columnType, ValueType>;
    friend ConstType;

    // default constructor
    SoAParametersImpl() = default;

    // constructor from individual address, stride and size
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl(ScalarType* addr, byte_size_type stride)
        : addr_(addr), stride_(stride) {}

    SOA_HOST_DEVICE SOA_INLINE ScalarType* data() const { return addr_; }
    SOA_HOST_DEVICE SOA_INLINE byte_size_type stride() const { return stride_; }

    // address, stride and size
    ScalarType* addr_ = nullptr;
    byte_size_type stride_ = 0;
  };

  // Matryoshka template to avoid commas inside macros
  template <SoAColumnType TColumnType>
  struct SoAParameters_ColumnType {
    template <typename T>
    using DataType = SoAParametersImpl<TColumnType, T>;
  };

  template <typename TColumn>
  struct Tuple {
    using Type = std::tuple<TColumn, size_type>;
  };

  template <typename TColumn>
  struct ConstTuple {
    using Type = std::tuple<typename TColumn::ConstType, size_type>;
  };

  // helper converting a const parameter set to a non-const parameter set, to be used only in the constructor of non-const "element"
  namespace {
    template <typename T>
    constexpr std::remove_const_t<T>* non_const_ptr(T* p) {
      return const_cast<std::remove_const_t<T>*>(p);
    }
  }  // namespace

  template <SoAColumnType TColumnType, typename T>
  SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl<TColumnType, T> const_cast_SoAParametersImpl(
      SoAConstParametersImpl<TColumnType, T> const& o) {
    return SoAParametersImpl<TColumnType, T>{non_const_ptr(o.addr_)};
  }

  template <typename T>
  SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl<SoAColumnType::eigen, T> const_cast_SoAParametersImpl(
      SoAConstParametersImpl<SoAColumnType::eigen, T> const& o) {
    return SoAParametersImpl<SoAColumnType::eigen, T>{non_const_ptr(o.addr_), o.stride_};
  }

  // Helper template managing the value at index idx within a column.
  // The optional compile time alignment parameter enables informing the
  // compiler of alignment (enforced by caller).
  template <SoAColumnType TColumnType,
            typename T,
            byte_size_type TAlignment,
            bool TRestrictQualify = restrict_qualify::disabled>
  class SoAValue {
    // eigen is implemented in a specialization
    static_assert(TColumnType != SoAColumnType::eigen);

  public:
    using restrict = add_restrict<T, TRestrictQualify>;
    using val = restrict ::value;
    using ptr = restrict ::pointer;
    using ref = restrict ::reference;
    using pointer_to_const = restrict ::pointer_to_const;
    using reference_to_const = restrict ::reference_to_const;

    SOA_HOST_DEVICE SOA_INLINE SoAValue(size_type i, T* col) : idx_(i), col_(col) {}

    SOA_HOST_DEVICE SOA_INLINE SoAValue(size_type i, SoAParametersImpl<TColumnType, T> params)
        : idx_(i), col_(params.addr_) {}

    SOA_HOST_DEVICE SOA_INLINE ref operator()() {
      // ptr type will add the restrict qualifier if needed
      ptr col = col_;
      return col[idx_];
    }

    SOA_HOST_DEVICE SOA_INLINE reference_to_const operator()() const {
      // pointer_to_const type will add the restrict qualifier if needed
      pointer_to_const col = col_;
      return col[idx_];
    }

    SOA_HOST_DEVICE SOA_INLINE ptr operator&() { return &col_[idx_]; }

    SOA_HOST_DEVICE SOA_INLINE pointer_to_const operator&() const { return &col_[idx_]; }

    using value_type = val;

    static constexpr auto value_size = sizeof(T);

  private:
    size_type idx_;
    T* col_;
  };

  // Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.
#ifdef EIGEN_WORLD_VERSION
  // Helper template managing an Eigen-type value at index idx within a column.
  template <class C, byte_size_type TAlignment, bool RestrictQualify>
  class SoAValue<SoAColumnType::eigen, C, TAlignment, RestrictQualify> {
  public:
    using type = C;
    using map_type = Eigen::Map<C, 0, Eigen::InnerStride<Eigen::Dynamic>>;
    using cmap_type = const Eigen::Map<const C, 0, Eigen::InnerStride<Eigen::Dynamic>>;
    using restrict = add_restrict<typename C::Scalar, RestrictQualify>;
    using val = typename restrict ::value;
    using ptr = typename restrict ::pointer;
    using ref = typename restrict ::reference;
    using pointer_to_const = typename restrict ::pointer_to_const;
    using reference_to_const = typename restrict ::reference_to_const;

    SOA_HOST_DEVICE SOA_INLINE SoAValue(size_type i, typename C::Scalar* col, byte_size_type stride)
        : val_(col + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
          crCol_(col),
          cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
          stride_(stride) {}

    SOA_HOST_DEVICE SOA_INLINE SoAValue(size_type i, SoAParametersImpl<SoAColumnType::eigen, C> params)
        : val_(params.addr_ + i,
               C::RowsAtCompileTime,
               C::ColsAtCompileTime,
               Eigen::InnerStride<Eigen::Dynamic>(params.stride_)),
          crCol_(params.addr_),
          cVal_(crCol_ + i,
                C::RowsAtCompileTime,
                C::ColsAtCompileTime,
                Eigen::InnerStride<Eigen::Dynamic>(params.stride_)),
          stride_(params.stride_) {}

    SOA_HOST_DEVICE SOA_INLINE map_type& operator()() { return val_; }

    SOA_HOST_DEVICE SOA_INLINE const cmap_type& operator()() const { return cVal_; }

    SOA_HOST_DEVICE SOA_INLINE operator C() { return val_; }

    SOA_HOST_DEVICE SOA_INLINE operator const C() const { return cVal_; }

    SOA_HOST_DEVICE SOA_INLINE C* operator&() { return &val_; }

    SOA_HOST_DEVICE SOA_INLINE const C* operator&() const { return &cVal_; }

    template <class C2>
    SOA_HOST_DEVICE SOA_INLINE map_type& operator=(const C2& v) {
      return val_ = v;
    }

    using value_type = typename C::Scalar;
    static constexpr auto value_size = sizeof(typename C::Scalar);
    SOA_HOST_DEVICE SOA_INLINE byte_size_type stride() const { return stride_; }

  private:
    map_type val_;
    const ptr crCol_;
    cmap_type cVal_;
    byte_size_type stride_;
  };
#else
  // Raise a compile-time error
  template <class C, byte_size_type TAlignment, bool TRestrictQualify>
  class SoAValue<SoAColumnType::eigen, C, TAlignment, TRestrictQualify> {
    static_assert(!sizeof(C),
                  "Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.");
  };
#endif

  // Matryoshka template to avoid commas inside macros
  template <SoAColumnType TColumnType>
  struct SoAValue_ColumnType {
    template <typename T>
    struct DataType {
      template <byte_size_type TAlignment>
      struct Alignment {
        template <bool TRestrictQualify>
        using Value = SoAValue<TColumnType, T, TAlignment, TRestrictQualify>;
      };
    };
  };

  // Helper template managing a const value at index idx within a column.
  template <SoAColumnType TColumnType,
            typename T,
            byte_size_type TAlignment,
            bool TRestrictQualify = restrict_qualify::disabled>
  class SoAConstValue {
    // Eigen is implemented in a specialization
    static_assert(TColumnType != SoAColumnType::eigen);

  public:
    using restrict = add_restrict<T, TRestrictQualify>;
    using val = restrict ::value;
    using ptr = restrict ::pointer;
    using ref = restrict ::reference;
    using pointer_to_const = restrict ::pointer_to_const;
    using reference_to_const = restrict ::reference_to_const;
    using Params = SoAParametersImpl<TColumnType, T>;
    using ConstParams = SoAConstParametersImpl<TColumnType, T>;

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, const T* col) : idx_(i), col_(col) {}

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, SoAParametersImpl<TColumnType, T> params)
        : idx_(i), col_(params.addr_) {}

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, SoAConstParametersImpl<TColumnType, T> params)
        : idx_(i), col_(params.addr_) {}

    SOA_HOST_DEVICE SOA_INLINE reference_to_const operator()() const {
      // ptr type will add the restrict qualifier if needed
      pointer_to_const col = col_;
      return col[idx_];
    }

    SOA_HOST_DEVICE SOA_INLINE const T* operator&() const { return &col_[idx_]; }

    using value_type = T;
    static constexpr auto value_size = sizeof(T);

  private:
    size_type idx_;
    const T* col_;
  };

  // Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.
#ifdef EIGEN_WORLD_VERSION
  // Helper template managing a const Eigen-type value at index idx within a column.
  template <class C, byte_size_type ALIGNMENT, bool RESTRICT_QUALIFY>
  class SoAConstValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
  public:
    using type = C;
    using cmap_type = Eigen::Map<const C, 0, Eigen::InnerStride<Eigen::Dynamic>>;
    using reference_to_const = const cmap_type&;
    using ConstParams = SoAConstParametersImpl<SoAColumnType::eigen, C>;

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, typename C::Scalar* col, byte_size_type stride)
        : crCol_(col),
          cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
          stride_(stride) {}

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, SoAConstParametersImpl<SoAColumnType::eigen, C> params)
        : crCol_(params.addr_),
          cVal_(crCol_ + i,
                C::RowsAtCompileTime,
                C::ColsAtCompileTime,
                Eigen::InnerStride<Eigen::Dynamic>(params.stride_)),
          stride_(params.stride_) {}

    SOA_HOST_DEVICE SOA_INLINE const cmap_type& operator()() const { return cVal_; }

    SOA_HOST_DEVICE SOA_INLINE operator const C() const { return cVal_; }

    SOA_HOST_DEVICE SOA_INLINE const C* operator&() const { return &cVal_; }

    using value_type = typename C::Scalar;
    static constexpr auto value_size = sizeof(typename C::Scalar);

    SOA_HOST_DEVICE SOA_INLINE byte_size_type stride() const { return stride_; }

  private:
    const typename C::Scalar* __restrict__ crCol_;
    cmap_type cVal_;
    byte_size_type stride_;
  };
#else
  // Raise a compile-time error
  template <class C, byte_size_type ALIGNMENT, bool RESTRICT_QUALIFY>
  class SoAConstValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
    static_assert(!sizeof(C),
                  "Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.");
  };
#endif

  // Matryoshka template to avoid commas inside macros
  template <SoAColumnType COLUMN_TYPE>
  struct SoAConstValue_ColumnType {
    template <typename T>
    struct DataType {
      template <byte_size_type ALIGNMENT>
      struct Alignment {
        template <bool RESTRICT_QUALIFY>
        using ConstValue = SoAConstValue<COLUMN_TYPE, T, ALIGNMENT, RESTRICT_QUALIFY>;
      };
    };
  };

  // Helper template to avoid commas inside macros
#ifdef EIGEN_WORLD_VERSION
  template <class C>
  struct EigenConstMapMaker {
    using Type = Eigen::Map<const C, Eigen::AlignmentType::Unaligned, Eigen::InnerStride<Eigen::Dynamic>>;

    class DataHolder {
    public:
      DataHolder(const typename C::Scalar* data) : data_(data) {}

      EigenConstMapMaker::Type withStride(byte_size_type stride) {
        return EigenConstMapMaker::Type(
            data_, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride));
      }

    private:
      const typename C::Scalar* const data_;
    };

    static DataHolder withData(const typename C::Scalar* data) { return DataHolder(data); }
  };
#else
  template <class C>
  struct EigenConstMapMaker {
    // Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.
    static_assert(!sizeof(C),
                  "Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.");
  };
#endif

  // helper function to compute aligned size
  // this is an integer division -> it rounds size to the next multiple of alignment
  constexpr byte_size_type alignSize(byte_size_type size, byte_size_type alignment) {
    assert(alignment > 0 && "Alignment for SoA must be > 0");
    return ((size + alignment - 1) / alignment) * alignment;
  }

}  // namespace ffx::soa

#define SOA_SCALAR(TYPE, NAME) (_VALUE_TYPE_SCALAR, TYPE, NAME, ~)
#define SOA_COLUMN(TYPE, NAME) (_VALUE_TYPE_COLUMN, TYPE, NAME, ~)
#define SOA_EIGEN_COLUMN(TYPE, NAME) (_VALUE_TYPE_EIGEN_COLUMN, TYPE, NAME, ~)
#define SOA_ELEMENT_METHODS(...) (_VALUE_TYPE_METHOD, _, _, (__VA_ARGS__))
#define SOA_CONST_ELEMENT_METHODS(...) (_VALUE_TYPE_CONST_METHOD, _, _, (__VA_ARGS__))
#define SOA_BLOCK(NAME, LAYOUT_NAME) (_VALUE_TYPE_BLOCK, NAME, LAYOUT_NAME)
#define SOA_VIEW_METHODS(...) (_VALUE_TYPE_VIEW_METHOD, _, (__VA_ARGS__))
#define SOA_CONST_VIEW_METHODS(...) (_VALUE_TYPE_CONST_VIEW_METHOD, _, (__VA_ARGS__))

/* Macro generating customized methods for the element */
#define GENERATE_METHODS(R, DATA, FIELD)                                         \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(0, FIELD), _VALUE_TYPE_METHOD), \
              BOOST_PP_TUPLE_ELEM(3, FIELD),                                     \
              BOOST_PP_EMPTY())

/* Macro generating customized methods for the const element*/
#define GENERATE_CONST_METHODS(R, DATA, FIELD)                                         \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(0, FIELD), _VALUE_TYPE_CONST_METHOD), \
              BOOST_PP_TUPLE_ELEM(3, FIELD),                                           \
              BOOST_PP_EMPTY())

/* Macro generating customized methods for the element */
#define GENERATE_VIEW_METHODS(R, DATA, FIELD)                                         \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(0, FIELD), _VALUE_TYPE_VIEW_METHOD), \
              BOOST_PP_TUPLE_ELEM(2, FIELD),                                          \
              BOOST_PP_EMPTY())

/* Macro generating customized methods for the const element*/
#define GENERATE_CONST_VIEW_METHODS(R, DATA, FIELD)                                         \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(0, FIELD), _VALUE_TYPE_CONST_VIEW_METHOD), \
              BOOST_PP_TUPLE_ELEM(2, FIELD),                                                \
              BOOST_PP_EMPTY())

/* Preprocessing loop for managing functions generation: only macros containing valid content are expanded */
#define ENUM_FOR_PRED(R, STATE) BOOST_PP_LESS(BOOST_PP_TUPLE_ELEM(0, STATE), BOOST_PP_TUPLE_ELEM(1, STATE))

#define ENUM_FOR_OP(R, STATE) \
  (BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(0, STATE)), BOOST_PP_TUPLE_ELEM(1, STATE), BOOST_PP_TUPLE_ELEM(2, STATE))

#define ENUM_FOR_MACRO(R, STATE) \
  BOOST_PP_TUPLE_ENUM(BOOST_PP_SEQ_ELEM(BOOST_PP_TUPLE_ELEM(0, STATE), BOOST_PP_TUPLE_ELEM(2, STATE)))

#define ENUM_IF_VALID(...)                                                                      \
  BOOST_PP_FOR((0, BOOST_PP_VARIADIC_SIZE(__VA_ARGS__), BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)), \
               ENUM_FOR_PRED,                                                                   \
               ENUM_FOR_OP,                                                                     \
               ENUM_FOR_MACRO)

/* Iterate on the macro MACRO and return the result as a comma separated list, converting
   the boost sequence into tuples and then into list */
#define _ITERATE_ON_ALL_COMMA(MACRO, DATA, ...) \
  BOOST_PP_TUPLE_ENUM(BOOST_PP_SEQ_TO_TUPLE(_ITERATE_ON_ALL(MACRO, DATA, __VA_ARGS__)))

/* Iterate MACRO on all elements of the boost sequence */
#define _ITERATE_ON_ALL(MACRO, DATA, ...) BOOST_PP_SEQ_FOR_EACH(MACRO, DATA, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

/* Switch on macros depending on scalar / column type */
#define _SWITCH_ON_TYPE(VALUE_TYPE, IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN) \
  BOOST_PP_IF(                                                             \
      BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_SCALAR),                      \
      IF_SCALAR,                                                           \
      BOOST_PP_IF(                                                         \
          BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_COLUMN),                  \
          IF_COLUMN,                                                       \
          BOOST_PP_IF(BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_EIGEN_COLUMN), IF_EIGEN_COLUMN, BOOST_PP_EMPTY())))

namespace ffx::soa {

  /* Column accessors: templates implementing the global accesors (soa::x() and soa::x(index) */
  enum class SoAAccessType : bool { mutableAccess, constAccess };

  template <typename, SoAColumnType, SoAAccessType, byte_size_type, bool>
  struct SoAColumnAccessorsImpl {};

  // Column
  template <typename T, byte_size_type TAlignment, bool TRestrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::column, SoAAccessType::mutableAccess, TAlignment, TRestrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::column, T>& params,
                                                      size_type size)
        : params_(params), size_(size) {}
    SOA_HOST_DEVICE SOA_INLINE std::span<T> operator()() { return std::span<T>(params_.addr_, size_); }

    using NoParamReturnType = std::span<T>;
    using ParamReturnType = T&;
    SOA_HOST_DEVICE SOA_INLINE T& operator()(size_type index) { return params_.addr_[index]; }

  private:
    SoAParametersImpl<SoAColumnType::column, T> params_;
    size_type size_ = 0;
  };

  // Const column
  template <typename T, byte_size_type TAlignment, bool TRestrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::column, SoAAccessType::constAccess, TAlignment, TRestrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::column, T>& params,
                                                      size_type size)
        : params_(params), size_(size) {}
    SOA_HOST_DEVICE SOA_INLINE std::span<const T> operator()() const {
      return std::span<const T>(params_.addr_, size_);
    }
    using NoParamReturnType = std::span<const T>;
    using ParamReturnType = const T&;
    SOA_HOST_DEVICE SOA_INLINE T const& operator()(size_type index) const { return params_.addr_[index]; }

  private:
    SoAConstParametersImpl<SoAColumnType::column, T> params_;
    const size_type size_ = 0;
  };

  // Scalar
  template <typename T, byte_size_type TAlignment, bool TRestrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::scalar, SoAAccessType::mutableAccess, TAlignment, TRestrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::scalar, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE SOA_INLINE T& operator()() { return *params_.addr_; }
    using NoParamReturnType = T&;
    using ParamReturnType = void;
    SOA_HOST_DEVICE SOA_INLINE void operator()(size_type index) const {
      assert(false && "Indexed access impossible for SoA scalars.");
    }

  private:
    SoAParametersImpl<SoAColumnType::scalar, T> params_;
  };

  // Const scalar
  template <typename T, byte_size_type TAlignment, bool TRestrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::scalar, SoAAccessType::constAccess, TAlignment, TRestrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::scalar, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE SOA_INLINE T const& operator()() const { return *params_.addr_; }
    using NoParamReturnType = T const&;
    using ParamReturnType = void;
    SOA_HOST_DEVICE SOA_INLINE void operator()(size_type index) const {
      assert(false && "Indexed access impossible for SoA scalars.");
    }

  private:
    SoAConstParametersImpl<SoAColumnType::scalar, T> params_;
  };

  // Eigen-type
  template <typename T, byte_size_type TAlignment, bool TRestrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::eigen, SoAAccessType::mutableAccess, TAlignment, TRestrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::eigen, T>& params,
                                                      size_type size)
        : params_(params), size_(size) {}
    SOA_HOST_DEVICE SOA_INLINE std::span<typename T::Scalar> operator()() {
      return std::span<typename T::Scalar>(params_.addr_, size_);
    }
    using NoParamReturnType = std::span<typename T::Scalar>;
    using ParamReturnType = SoAValue<SoAColumnType::eigen, T, TAlignment, TRestrictQualify>::map_type;
    SOA_HOST_DEVICE SOA_INLINE ParamReturnType operator()(size_type index) {
      return SoAValue<SoAColumnType::eigen, T, TAlignment, TRestrictQualify>(index, params_)();
    }

  private:
    SoAParametersImpl<SoAColumnType::eigen, T> params_;
    size_type size_ = 0;
  };

  // Const Eigen-type
  template <typename T, byte_size_type TAlignment, bool TRestrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::eigen, SoAAccessType::constAccess, TAlignment, TRestrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::eigen, T>& params,
                                                      size_type size)
        : params_(params), size_(size) {}
    SOA_HOST_DEVICE SOA_INLINE std::span<typename T::Scalar const> operator()() const {
      return std::span<typename T::Scalar const>(params_.addr_, size_);
    }
    using NoParamReturnType = std::span<typename T::Scalar const>;
    using ParamReturnType = SoAValue<SoAColumnType::eigen, T, TAlignment, TRestrictQualify>::cmap_type;
    SOA_HOST_DEVICE SOA_INLINE ParamReturnType operator()(size_type index) const {
      return SoAConstValue<SoAColumnType::eigen, T, TAlignment, TRestrictQualify>(index, params_)();
    }

  private:
    SoAConstParametersImpl<SoAColumnType::eigen, T> params_;
    const size_type size_ = 0;
  };

  /* A helper template stager to avoid commas inside macros */
  template <typename T>
  struct SoAAccessors {
    template <auto TColumnType>
    struct ColumnType {
      template <auto TAccessType>
      struct AccessType {
        template <auto TAlignment>
        struct Alignment {
          template <auto TRestrictQualify>
          struct RestrictQualifier
              : public SoAColumnAccessorsImpl<T, TColumnType, TAccessType, TAlignment, TRestrictQualify> {
            using SoAColumnAccessorsImpl<T, TColumnType, TAccessType, TAlignment, TRestrictQualify>::
                SoAColumnAccessorsImpl;
          };
        };
      };
    };
  };

  /* Enum parameters allowing templated control of layout/view behaviors */
  /* Alignment enforcement verifies every column is aligned, and
   * hints the compiler that it can expect column pointers to be aligned */
  struct AlignmentEnforcement {
    static constexpr bool relaxed = false;
    static constexpr bool enforced = true;
  };

  struct CacheLineSize {
    // Nvidia, AMD GPU cache line / transaction size
    static constexpr byte_size_type Gpu = 128;

    // Intel, AMD, ARM CPU cache line size
    static constexpr byte_size_type Cpu =
#ifdef ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT
        ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT;
#else
        64;
#endif  // ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT

    static constexpr byte_size_type defaultSize =
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        Gpu;
#else
        Cpu;
#endif
  };

}  // namespace ffx::soa

// Small wrapper for stream insertion of SoA printing
template <
    typename SOA,
    typename TSfinae = std::enable_if_t<std::is_invocable_v<decltype(&SOA::soaToStreamInternal), SOA&, std::ostream&>>>
SOA_HOST_ONLY std::ostream& operator<<(std::ostream& os, const SOA& soa) {
  soa.soaToStreamInternal(os);
  return os;
}

namespace ffx::soa::detail {

  template <range_checking::Mode M>
  struct IndexWithSourceLocation {
    static constexpr auto mode = M;

    SOA_HOST_DEVICE constexpr IndexWithSourceLocation(size_type value) noexcept : value_{value} {}

    size_type value_;
  };

  template <range_checking::Mode M>
    requires(M == range_checking::extended)
  struct IndexWithSourceLocation<M> {
    static constexpr auto mode = M;

    SOA_HOST_DEVICE constexpr IndexWithSourceLocation(
        size_type value, std::source_location location = std::source_location::current()) noexcept
        : value_{value}, location_{location} {}

    size_type value_;
    std::source_location location_;
  };

  template <range_checking::Mode M>
  [[noreturn]] void throwOutOfRangeError(const char* message, const IndexWithSourceLocation<M>& index, size_type range);

  template <>
  [[noreturn]] void throwOutOfRangeError<range_checking::extended>(
      const char* message, const IndexWithSourceLocation<range_checking::extended>& index, size_type range) {
    throw std::out_of_range(std::format("{}: index {} out of range {} at file {} at line {}\n",
                                        message,
                                        index.value_,
                                        range,
                                        index.location_.file_name(),
                                        index.location_.line()));
  }

  template <>
  [[noreturn]] void throwOutOfRangeError<range_checking::enabled>(
      const char* message, const IndexWithSourceLocation<range_checking::enabled>& index, size_type range) {
    throw std::out_of_range(std::format("{}: index {} out of range {}\n", message, index.value_, range));
  }

  template <>
  [[noreturn]] void throwOutOfRangeError<range_checking::disabled>(
      const char* message, const IndexWithSourceLocation<range_checking::disabled>& index, size_type range) {
    throw std::out_of_range(std::format("{}: index {} out of range {}\n", message, index.value_, range));
  }

  // helper function to check alignment of a pointer. Returns true if the pointer is not aligned to the specified alignment.
  [[noreturn]] void throwRuntimeError(const char* message) { throw std::runtime_error(message); }

  template <typename T>
  SOA_INLINE void checkAlignment(const T* addr, byte_size_type alignment, const char* message) {
    if (reinterpret_cast<uintptr_t>(addr) % alignment)
      throwRuntimeError(message);
  }

  template <typename ColumnType>
  struct PrintColumn;

  // Helper struct for streaming columns
  template <typename T>
  struct PrintColumn<SoAParametersImpl<SoAColumnType::scalar, T>> {
    void operator()(std::ostream& soa_impl_os,
                    std::string_view name,
                    byte_size_type& soa_impl_offset,
                    size_type,
                    byte_size_type alignment) {
      const auto size = sizeof(T);
      soa_impl_os << " Scalar " << name << " at offset " << soa_impl_offset << " has size " << size << " and padding "
                  << alignSize(size, alignment) - size << std::endl;
      soa_impl_offset += alignSize(size, alignment);
    }
  };

  template <typename T>
  struct PrintColumn<SoAParametersImpl<SoAColumnType::column, T>> {
    void operator()(std::ostream& soa_impl_os,
                    std::string_view name,
                    byte_size_type& soa_impl_offset,
                    size_type elements,
                    byte_size_type alignment) {
      const auto size = sizeof(T) * elements;
      soa_impl_os << " Column " << name << " at offset " << soa_impl_offset << " has size " << size << " and padding "
                  << alignSize(size, alignment) - size << std::endl;
      soa_impl_offset += alignSize(size, alignment);
    }
  };

  template <typename T>
  struct PrintColumn<SoAParametersImpl<SoAColumnType::eigen, T>> {
    void operator()(std::ostream& soa_impl_os,
                    std::string_view name,
                    byte_size_type& soa_impl_offset,
                    size_type elements,
                    byte_size_type alignment) {
      const auto size = elements * sizeof(typename T::Scalar);
      soa_impl_os << " Eigen value " << name << " at offset " << soa_impl_offset << " has dimension " << "("
                  << T::RowsAtCompileTime << " x " << T::ColsAtCompileTime << ")" << " and per column size " << size
                  << " and padding " << alignSize(size, alignment) - size << std::endl;
      soa_impl_offset += alignSize(size, alignment) * T::RowsAtCompileTime * T::ColsAtCompileTime;
    }
  };

  // Helper struct for computing the pitch of each column
  template <typename ColumnType>
  struct ComputePitch;

  template <typename T>
  struct ComputePitch<SoAParametersImpl<SoAColumnType::scalar, T>> {
    SOA_HOST_DEVICE constexpr byte_size_type operator()(size_type, byte_size_type alignment) const {
      return alignSize(sizeof(T), alignment);
    }
  };

  template <typename T>
  struct ComputePitch<SoAParametersImpl<SoAColumnType::column, T>> {
    SOA_HOST_DEVICE constexpr byte_size_type operator()(size_type elements, byte_size_type alignment) const {
      return alignSize(elements * sizeof(T), alignment);
    }
  };

  template <typename T>
  struct ComputePitch<SoAParametersImpl<SoAColumnType::eigen, T>> {
    SOA_HOST_DEVICE constexpr byte_size_type operator()(size_type elements, byte_size_type alignment) const {
      return alignSize(elements * sizeof(typename T::Scalar), alignment) * T::RowsAtCompileTime * T::ColsAtCompileTime;
    }
  };

  // Helper type trait for obtaining a span type for a column
  template <typename ColumnType>
  struct GetSpanType;

  template <typename T>
  struct GetSpanType<SoAConstParametersImpl<SoAColumnType::scalar, T>> {
    using type = std::span<T, 1>;
  };

  template <typename T>
  struct GetSpanType<SoAConstParametersImpl<SoAColumnType::column, T>> {
    using type = std::span<T>;
  };

  template <typename T>
  struct GetSpanType<SoAConstParametersImpl<SoAColumnType::eigen, T>> {
    using type = std::span<typename T::Scalar>;
  };

  template <typename T>
  struct GetSpanType<SoAParametersImpl<SoAColumnType::scalar, T>> {
    using type = std::span<T, 1>;
  };

  template <typename T>
  struct GetSpanType<SoAParametersImpl<SoAColumnType::column, T>> {
    using type = std::span<T>;
  };

  template <typename T>
  struct GetSpanType<SoAParametersImpl<SoAColumnType::eigen, T>> {
    using type = std::span<typename T::Scalar>;
  };

  template <typename ColumnType>
  using SpanType = GetSpanType<ColumnType>::type;

  // Helper type trait for obtaining a const-span type for a column
  template <typename ColumnType>
  struct GetConstSpanType;

  template <typename T>
  struct GetConstSpanType<SoAConstParametersImpl<SoAColumnType::scalar, T>> {
    using type = std::span<std::add_const_t<T>, 1>;
  };

  template <typename T>
  struct GetConstSpanType<SoAConstParametersImpl<SoAColumnType::column, T>> {
    using type = std::span<std::add_const_t<T>>;
  };

  template <typename T>
  struct GetConstSpanType<SoAConstParametersImpl<SoAColumnType::eigen, T>> {
    using type = std::span<std::add_const_t<typename T::Scalar>>;
  };

  template <typename T>
  struct GetConstSpanType<SoAParametersImpl<SoAColumnType::scalar, T>> {
    using type = std::span<std::add_const_t<T>, 1>;
  };

  template <typename T>
  struct GetConstSpanType<SoAParametersImpl<SoAColumnType::column, T>> {
    using type = std::span<std::add_const_t<T>>;
  };

  template <typename T>
  struct GetConstSpanType<SoAParametersImpl<SoAColumnType::eigen, T>> {
    using type = std::span<std::add_const_t<typename T::Scalar>>;
  };

  template <typename ColumnType>
  using ConstSpanType = GetConstSpanType<ColumnType>::type;

  // Helper functions for constructing a span from a column
  template <typename T>
  auto getSpanToColumn(const SoAParametersImpl<SoAColumnType::scalar, T>& column, size_type, byte_size_type alignment) {
    return std::span(column.addr_, 1);
  }

  template <typename T>
  auto getSpanToColumn(const SoAParametersImpl<SoAColumnType::column, T>& column,
                       size_type elements,
                       byte_size_type alignment) {
    return std::span(column.addr_, elements);
  }

  template <typename T>
  auto getSpanToColumn(const SoAParametersImpl<SoAColumnType::eigen, T>& column,
                       size_type elements,
                       byte_size_type alignment) {
    return std::span(column.addr_,
                     alignSize(elements * sizeof(typename T::Scalar), alignment) * T::RowsAtCompileTime *
                         T::ColsAtCompileTime / sizeof(typename T::Scalar));
  }

  template <typename T>
  auto getSpanToColumn(const SoAConstParametersImpl<SoAColumnType::scalar, T>& column,
                       size_type elements,
                       byte_size_type alignment) {
    return std::span(column.addr_, 1);
  }

  template <typename T>
  auto getSpanToColumn(const SoAConstParametersImpl<SoAColumnType::column, T>& column,
                       size_type elements,
                       byte_size_type alignment) {
    return std::span(column.addr_, elements);
  }

  template <typename T>
  auto getSpanToColumn(const SoAConstParametersImpl<SoAColumnType::eigen, T>& column,
                       size_type elements,
                       byte_size_type alignment) {
    return std::span(column.addr_,
                     alignSize(elements * sizeof(typename T::Scalar), alignment) * T::RowsAtCompileTime *
                         T::ColsAtCompileTime / sizeof(typename T::Scalar));
  }

  // Helper function for extracting the number of blocks of a layout. Falls back to 1 if the layout does not define a static member blocksNumber.
  template <typename T>
  constexpr size_type nBlocks() {
    if constexpr (requires { T::blocksNumber; })
      return T::blocksNumber;
    else
      return static_cast<size_type>(1);
  }

  // Case 1: type has blocksNumber → returns a sub-array
  template <typename T, size_type N>
    requires requires { T::blocksNumber; }
  [[nodiscard]] constexpr std::array<size_type, T::blocksNumber> extractSegment(const std::array<size_type, N>& sizes,
                                                                                size_type offset) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return std::array<size_type, T::blocksNumber>{sizes[offset + I]...};
    }(std::make_index_sequence<T::blocksNumber>{});
  }

  // Case 2: fallback (single block) → returns a scalar
  template <typename T, size_type N>
    requires(!requires { T::blocksNumber; })
  [[nodiscard]] constexpr size_type extractSegment(const std::array<size_type, N>& sizes, size_type offset) {
    return sizes[offset];
  }

}  // namespace ffx::soa::detail

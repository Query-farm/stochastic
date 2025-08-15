#include <type_traits>
#include <tuple>

// Primary template
template <typename T>
struct callable_traits;

// --- Function pointer ---
template <typename Ret, typename... Args>
struct callable_traits<Ret (*)(Args...)> {
	using return_t = Ret;
	static constexpr std::size_t arity = sizeof...(Args);
	template <std::size_t N>
	using arg_t = std::tuple_element_t<N, std::tuple<Args...>>;
};

// --- Function reference ---
template <typename Ret, typename... Args>
struct callable_traits<Ret (&)(Args...)> : callable_traits<Ret (*)(Args...)> {};

// --- Member function pointer ---
template <typename ClassType, typename Ret, typename... Args>
struct callable_traits<Ret (ClassType::*)(Args...) const> {
	using return_t = Ret;
	static constexpr std::size_t arity = sizeof...(Args);
	template <std::size_t N>
	using arg_t = std::tuple_element_t<N, std::tuple<Args...>>;
};

template <typename ClassType, typename Ret, typename... Args>
struct callable_traits<Ret (ClassType::*)(Args...)> {
	using return_t = Ret;
	static constexpr std::size_t arity = sizeof...(Args);
	template <std::size_t N>
	using arg_t = std::tuple_element_t<N, std::tuple<Args...>>;
};

// --- Lambdas / functors fallback ---
template <typename T>
struct callable_traits : callable_traits<decltype(&T::operator())> {};

template <typename>
struct AlwaysFalse : std::false_type {};

template <typename T>
constexpr bool AlwaysFalse_v = AlwaysFalse<T>::value;
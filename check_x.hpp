#pragma once

#include <cstdio>

#include <stdexcept>
#include <string_view>
#include <vector>

constexpr const char* get_filename(const char* filename_abs) {
    size_t const pos = std::string_view(filename_abs).rfind("/");
    return (pos != std::string_view::npos) ? &filename_abs[pos+1] : filename_abs;
}

template <typename Func>
class Defer {
public:
    Defer(Func func) : func_(func) {}
    ~Defer() { this->func_(); }
private:
    Func func_;
};

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a ## b
#define UNIQUE_NAME(base) CONCAT(base, __LINE__)

#define DEFER_CODE(code) Defer UNIQUE_NAME(defer_)([&]()code)

#define DEFER_FUNC(func, ...) DEFER_CODE({func(__VA_ARGS__);})

template <typename Func>
auto check_zero(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err != 0)
    {
        std::vector<char> strbuf(1);

        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%d\n", filename, lineno, funcname, err); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
    return err;
}

#define CHECK_ZERO(func, ...) check_zero(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

#define DEFER_CHECK_ZERO(func, ...) DEFER_CODE({CHECK_ZERO(func, __VA_ARGS__);})


template <typename Func>
auto check_nonzero(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err == 0)
    {
        std::vector<char> strbuf(1);

        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%d\n", filename, lineno, funcname, err); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
    return err;
}

#define CHECK_NONZERO(func, ...) check_nonzero(get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

#define DEFER_CHECK_NONZERO(func, ...) DEFER_CODE({CHECK_NONZERO(func, __VA_ARGS__);})


template <typename CheckerFunc, typename Func>
auto check_value(char const* const filename, int const lineno, CheckerFunc checker_func, char const* const funcname, Func func)
{
    auto value = func();
    auto err = checker_func(value);
    if (err != 0)
    {
        std::vector<char> strbuf(1);

        auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%d\n", filename, lineno, funcname, err); };

        int str_length = printf_lambda(strbuf.data(), strbuf.size());
        strbuf.resize(str_length + 1);
        str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

        throw std::runtime_error(strbuf.data());
    }
    return value;
}


#define CHECK_VALUE(checker_code, func, ...) check_value(get_filename(__FILE__), __LINE__, [&](auto arg)checker_code, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})


//
// Created by windy on 2023/8/31.
//

#ifndef CUDALEARNING_UTILS_H
#define CUDALEARNING_UTILS_H


#include <chrono>
#include <functional>

using namespace std::chrono;

struct cuComplex {
    float   r;
    float   i;
    cuComplex( float a, float b ) : r(a), i(b)  {}
    [[nodiscard]] float magnitude2() const { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a) const {
        return {r*a.r - i*a.i, i*a.r + r*a.i};
    }
    cuComplex operator+(const cuComplex& a) const {
        return {r+a.r, i+a.i};
    }
};

template <typename Ret>
Ret get_executing_time(const std::function<void()>& inner_function) {
    auto start = system_clock::now().time_since_epoch();
    inner_function();
    auto end = system_clock::now().time_since_epoch();

    auto dura = duration_cast<duration<Ret, std::milli>>(end - start);
    return dura.count();
}

#endif //CUDALEARNING_UTILS_H

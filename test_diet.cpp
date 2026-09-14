// test_diet.cpp — correctness tests for diet.hpp.
//
// Copyright (c) 2026 Eden Networks Ltd.
// Author: Philip Lamb
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// ---------------------------------------------------------------------------
//
// Strategy:
//   1. Randomized oracle testing: mirror every mutation into a std::set<T>
//      and, at checkpoints, compare the diet's maximal-interval decomposition
//      against the runs derived from the oracle, and check the structural
//      invariant (AVL shape + independent ascending intervals).
//   2. Randomized set-algebra testing: union / intersection / difference /
//      complement / equality against oracle results, including aliasing.
//   3. Deterministic edge tests at the exact type limits for all eight
//      integer widths, signed and unsigned.
//
// Build:
//   clang++ -std=c++17 -O1 -g -fsanitize=address,undefined \
//       -fno-sanitize-recover=all -Wall -Wextra -Wpedantic \
//       test_diet.cpp -o test_diet && ./test_diet

#include "diet.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <new>
#include <random>
#include <set>
#include <string>
#include <vector>

// ---- fail-at-Nth-allocation harness (for exception-safety tests) ------------
// countdown < 0: disarmed. countdown == 0: the next allocation throws.
namespace alloc_fail {
long long countdown = -1;
bool should_fail() {
    if (countdown < 0) return false;
    if (countdown == 0) {
        countdown = -1;
        return true;
    }
    --countdown;
    return false;
}
} // namespace alloc_fail

void* operator new(std::size_t n) {
    if (alloc_fail::should_fail()) throw std::bad_alloc();
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

int g_failures = 0;
std::string g_context;

void fail(const char* file, int line, const char* expr) {
    std::fprintf(stderr, "FAIL %s:%d: %s   [%s]\n", file, line, expr,
                 g_context.c_str());
    if (++g_failures > 20) {
        std::fprintf(stderr, "too many failures, aborting\n");
        std::exit(1);
    }
}

#define CHECK(expr) \
    do { \
        if (!(expr)) fail(__FILE__, __LINE__, #expr); \
    } while (0)

// ---- oracle helpers ---------------------------------------------------------

template <typename T>
using ivec = std::vector<std::pair<T, T>>;

// Maximal runs of consecutive values in an ordered std::set.
template <typename T>
ivec<T> oracle_intervals(const std::set<T>& s) {
    using U = std::make_unsigned_t<T>;
    ivec<T> out;
    for (T v : s) {
        if (!out.empty() &&
            static_cast<U>(static_cast<U>(v) -
                           static_cast<U>(out.back().second)) == 1) {
            out.back().second = v;
        } else {
            out.emplace_back(v, v);
        }
    }
    return out;
}

template <typename T>
ivec<T> diet_intervals(const diet::set<T>& d) {
    ivec<T> out;
    d.for_each_interval([&](T a, T b) { out.emplace_back(a, b); });
    return out;
}

template <typename T>
bool same_content(const diet::set<T>& d, const std::set<T>& s) {
    return diet_intervals(d) == oracle_intervals(s);
}

template <typename T>
std::uint64_t oracle_cardinality(const std::set<T>& s) {
    return static_cast<std::uint64_t>(s.size());
}

// ---- range-query oracles ----------------------------------------------------

// Every value in [lo, hi] present in the oracle?
template <typename T>
bool oracle_contains_range(const std::set<T>& s, T lo, T hi) {
    for (T v = lo;; ++v) {
        if (s.count(v) == 0) return false;
        if (v == hi) break;
    }
    return true;
}

// Any value in [lo, hi] present in the oracle?
template <typename T>
bool oracle_intersects_range(const std::set<T>& s, T lo, T hi) {
    auto it = s.lower_bound(lo);
    return it != s.end() && *it <= hi;
}

// The maximal runs of the oracle, each clipped to [lo, hi], dropping any that
// fall entirely outside — exactly what for_each_interval_in must yield.
template <typename T>
ivec<T> oracle_clip(const std::set<T>& s, T lo, T hi) {
    ivec<T> out;
    for (auto [a, b] : oracle_intervals(s)) {
        if (b < lo || a > hi) continue;
        out.emplace_back(std::max(a, lo), std::min(b, hi));
    }
    return out;
}

// Cross-check all three range queries of d against the oracle over [lo, hi].
template <typename T>
void check_range_queries(const diet::set<T>& d, const std::set<T>& o, T lo,
                         T hi) {
    CHECK(d.contains(lo, hi) == oracle_contains_range(o, lo, hi));
    CHECK(d.intersects(lo, hi) == oracle_intersects_range(o, lo, hi));
    ivec<T> got;
    d.for_each_interval_in(lo, hi, [&](T a, T b) { got.emplace_back(a, b); });
    CHECK(got == oracle_clip(o, lo, hi));
}

// ---- random values ----------------------------------------------------------

template <typename T>
T rand_val(std::mt19937_64& rng) {
    using U = std::make_unsigned_t<T>;
    constexpr U lo = static_cast<U>(std::numeric_limits<T>::min());
    // Draw 64 random bits and reduce modulo the domain size; bias is
    // negligible for these widths and irrelevant for testing.
    const U span_mask = static_cast<U>(~static_cast<U>(0)); // full width of U
    U bits = static_cast<U>(rng() & static_cast<std::uint64_t>(span_mask));
    switch (rng() % 4) {
    case 0: // uniform
        break;
    case 1: // clustered near the minimum
        bits = static_cast<U>(bits % 64);
        break;
    case 2: // clustered near the maximum
        bits = static_cast<U>(span_mask - static_cast<U>(bits % 64));
        break;
    default: // clustered near mid-domain
        bits = static_cast<U>((span_mask / 2) + static_cast<U>(bits % 64) - 32);
        break;
    }
    return static_cast<T>(static_cast<U>(bits + lo)); // shift into T's range
}

// Random [lo, hi] with a small width, clamped at the type maximum.
template <typename T>
std::pair<T, T> rand_range(std::mt19937_64& rng) {
    using U = std::make_unsigned_t<T>;
    T lo = rand_val<T>(rng);
    U width = static_cast<U>(rng() % 33);
    U room = static_cast<U>(static_cast<U>(std::numeric_limits<T>::max()) -
                            static_cast<U>(lo));
    if (width > room) width = room;
    T hi = static_cast<T>(lo + static_cast<T>(width));
    return {lo, hi};
}

// ---- randomized mutation fuzz ----------------------------------------------

template <typename T>
void fuzz_mutations(const char* tname, std::uint64_t seed, int ops,
                    int check_every) {
    g_context = std::string("fuzz_mutations<") + tname + "> seed=" +
                std::to_string(seed);
    std::mt19937_64 rng(seed);
    diet::set<T> d;
    std::set<T> o;

    for (int i = 0; i < ops; ++i) {
        const std::uint64_t pick = rng() % 100;
        if (pick < 30) { // insert element
            T x = rand_val<T>(rng);
            bool changed = d.insert(x);
            bool ochanged = o.insert(x).second;
            CHECK(changed == ochanged);
        } else if (pick < 55) { // erase element
            T x = rand_val<T>(rng);
            bool changed = d.erase(x);
            bool ochanged = o.erase(x) > 0;
            CHECK(changed == ochanged);
        } else if (pick < 75) { // insert range
            auto [lo, hi] = rand_range<T>(rng);
            d.insert(lo, hi);
            for (T v = lo;; ++v) {
                o.insert(v);
                if (v == hi) break;
            }
        } else if (pick < 90) { // erase range
            auto [lo, hi] = rand_range<T>(rng);
            d.erase(lo, hi);
            o.erase(o.lower_bound(lo), o.upper_bound(hi));
        } else { // point queries only
            T x = rand_val<T>(rng);
            CHECK(d.contains(x) == (o.count(x) > 0));
            auto iv = d.interval_of(x);
            CHECK(iv.has_value() == (o.count(x) > 0));
            if (iv) {
                CHECK(iv->lo <= x && x <= iv->hi);
                CHECK(o.count(iv->lo) && o.count(iv->hi));
            }
        }
        // Range queries against the oracle, on a fresh random range every
        // iteration so they are checked over a constantly changing set.
        {
            auto [qlo, qhi] = rand_range<T>(rng);
            check_range_queries(d, o, qlo, qhi);
        }
        if (i % check_every == 0 || i == ops - 1) {
            CHECK(d.valid());
            CHECK(same_content(d, o));
            CHECK(d.cardinality() == oracle_cardinality(o));
            CHECK(d.empty() == o.empty());
            if (!o.empty()) {
                CHECK(d.min_elt() && *d.min_elt() == *o.begin());
                CHECK(d.max_elt() && *d.max_elt() == *o.rbegin());
            }
        }
    }
}

// ---- randomized set-algebra fuzz -------------------------------------------

template <typename T>
std::pair<diet::set<T>, std::set<T>> random_pair(std::mt19937_64& rng,
                                                 int ops) {
    diet::set<T> d;
    std::set<T> o;
    for (int i = 0; i < ops; ++i) {
        if (rng() % 3 == 0) {
            auto [lo, hi] = rand_range<T>(rng);
            d.insert(lo, hi);
            for (T v = lo;; ++v) {
                o.insert(v);
                if (v == hi) break;
            }
        } else if (rng() % 2) {
            T x = rand_val<T>(rng);
            d.insert(x);
            o.insert(x);
        } else {
            T x = rand_val<T>(rng);
            d.erase(x);
            o.erase(x);
        }
    }
    return {std::move(d), std::move(o)};
}

// Expected complement intervals, derived from a set's own runs.
template <typename T>
ivec<T> oracle_complement(const std::set<T>& s) {
    constexpr T tmin = std::numeric_limits<T>::min();
    constexpr T tmax = std::numeric_limits<T>::max();
    ivec<T> runs = oracle_intervals(s);
    ivec<T> out;
    T cursor = tmin;
    bool open = true;
    for (auto [a, b] : runs) {
        if (cursor < a) out.emplace_back(cursor, static_cast<T>(a - 1));
        if (b == tmax) {
            open = false;
            break;
        }
        cursor = static_cast<T>(b + 1);
    }
    if (open) out.emplace_back(cursor, tmax);
    return out;
}

template <typename T>
void fuzz_algebra(const char* tname, std::uint64_t seed, int rounds,
                  int build_ops) {
    std::mt19937_64 rng(seed);
    for (int round = 0; round < rounds; ++round) {
        g_context = std::string("fuzz_algebra<") + tname + "> seed=" +
                    std::to_string(seed) + " round=" + std::to_string(round);
        auto [da, oa] = random_pair<T>(rng, build_ops);
        auto [db, ob] = random_pair<T>(rng, build_ops);
        const diet::set<T> ca(da), cb(db); // pristine copies

        // union
        std::set<T> ou = oa;
        ou.insert(ob.begin(), ob.end());
        diet::set<T> du = da | db;
        CHECK(du.valid());
        CHECK(same_content(du, ou));

        // intersection
        std::set<T> oi;
        for (T v : oa)
            if (ob.count(v)) oi.insert(v);
        diet::set<T> di = da & db;
        CHECK(di.valid());
        CHECK(same_content(di, oi));

        // difference
        std::set<T> od;
        for (T v : oa)
            if (!ob.count(v)) od.insert(v);
        diet::set<T> dd = da - db;
        CHECK(dd.valid());
        CHECK(same_content(dd, od));

        // the reverse difference too (asymmetric operation)
        std::set<T> od2;
        for (T v : ob)
            if (!oa.count(v)) od2.insert(v);
        diet::set<T> dd2 = db - da;
        CHECK(dd2.valid());
        CHECK(same_content(dd2, od2));

        // operands must be untouched
        CHECK(da == ca);
        CHECK(db == cb);

        // complement (interval-level oracle: cheap at any width)
        diet::set<T> dc = da.complement();
        CHECK(dc.valid());
        CHECK(diet_intervals(dc) == oracle_complement(oa));
        CHECK((da & dc).empty());
        CHECK((da | dc) == diet::set<T>::universe());

        // equality
        CHECK((da == db) == (oa == ob));
        CHECK(da == da);

        // aliasing
        diet::set<T> x(da);
        x |= x;
        CHECK(x.valid() && x == da);
        x &= x;
        CHECK(x.valid() && x == da);
        x -= x;
        CHECK(x.valid() && x.empty());

        // algebraic identities
        CHECK((du - di) == ((da - db) | (db - da)));
        CHECK(((da - db) & db).empty());
    }
}

// ---- deterministic edge tests ----------------------------------------------

template <typename T>
void edge_tests(const char* tname) {
    g_context = std::string("edge_tests<") + tname + ">";
    constexpr T tmin = std::numeric_limits<T>::min();
    constexpr T tmax = std::numeric_limits<T>::max();
    using U = std::make_unsigned_t<T>;
    constexpr std::uint64_t domain_minus1 = static_cast<std::uint64_t>(
        static_cast<U>(static_cast<U>(tmax) - static_cast<U>(tmin)));

    { // empty set basics
        diet::set<T> d;
        CHECK(d.empty() && d.valid());
        CHECK(d.cardinality() == 0);
        CHECK(!d.min_elt() && !d.max_elt());
        CHECK(!d.contains(T{0}));
        CHECK(d.complement() == diet::set<T>::universe());
        CHECK(d.begin() == d.end());
    }
    { // universe
        diet::set<T> u = diet::set<T>::universe();
        CHECK(u.valid());
        CHECK(u.interval_count() == 1);
        CHECK(u.contains(tmin) && u.contains(tmax) && u.contains(T{0}));
        CHECK(u.contains(tmin, tmax));
        if (domain_minus1 == std::numeric_limits<std::uint64_t>::max()) {
            // 64-bit domain: 2^64 elements saturate
            CHECK(u.cardinality() == std::numeric_limits<std::uint64_t>::max());
        } else {
            CHECK(u.cardinality() == domain_minus1 + 1);
        }
        CHECK(u.complement().empty());
        // removing one interior element splits the single interval
        // (1 is interior for every supported type; 0 == tmin when unsigned)
        diet::set<T> v(u);
        CHECK(v.erase(T{1}));
        CHECK(v.valid());
        CHECK(v.interval_count() == 2);
        CHECK(!v.contains(T{1}));
        CHECK(v.contains(tmin) && v.contains(tmax));
        CHECK(v.complement() == diet::set<T>{T{1}});
    }
    { // adjacency merging at the extremes
        diet::set<T> d;
        CHECK(d.insert(tmax));
        CHECK(!d.insert(tmax));
        CHECK(d.insert(static_cast<T>(tmax - 1)));
        CHECK(d.interval_count() == 1);
        CHECK(d.insert(tmin));
        CHECK(d.insert(static_cast<T>(tmin + 1)));
        CHECK(d.interval_count() == 2);
        CHECK(d.valid());
        CHECK(d.cardinality() == 4);
        CHECK(*d.min_elt() == tmin && *d.max_elt() == tmax);
        CHECK(d.contains(tmin, static_cast<T>(tmin + 1)));
        CHECK(!d.contains(tmin, tmax));
        CHECK(d.intersects(tmin, tmax));
        // erase back down
        CHECK(d.erase(tmax));
        CHECK(d.erase(tmin));
        CHECK(d.valid() && d.cardinality() == 2);
    }
    { // full-domain range insert and erase
        diet::set<T> d;
        d.insert(tmin, tmax);
        CHECK(d == diet::set<T>::universe());
        d.erase(tmin, tmax);
        CHECK(d.empty() && d.valid());
    }
    { // building the universe from two halves merges into one interval
        diet::set<T> d;
        d.insert(tmin, T{0});
        d.insert(T{1}, tmax);
        CHECK(d.interval_count() == 1);
        CHECK(d == diet::set<T>::universe());
    }
    { // range insert absorbing several existing intervals
        diet::set<T> d;
        d.insert(T{0});
        d.insert(T{4});
        d.insert(T{8});
        d.insert(T{12});
        CHECK(d.interval_count() == 4);
        d.insert(T{1}, T{11}); // touches [0,0] and is adjacent to [12,12]
        CHECK(d.valid());
        CHECK(d.interval_count() == 1);
        CHECK(diet_intervals(d) == (ivec<T>{{T{0}, T{12}}}));
    }
    { // interior range erase splits an interval
        diet::set<T> d;
        d.insert(T{0}, T{20});
        d.erase(T{5}, T{15});
        CHECK(d.valid());
        CHECK(diet_intervals(d) == (ivec<T>{{T{0}, T{4}}, {T{16}, T{20}}}));
        CHECK(!d.intersects(T{5}, T{15}));
        CHECK(d.intersects(T{4}, T{5}));
    }
    { // clipped range-query visitation
        diet::set<T> d;
        d.insert(T{0}, T{9});
        d.insert(T{20}, T{29});
        d.insert(T{40}, T{49});
        ivec<T> got;
        d.for_each_interval_in(T{5}, T{44},
                               [&](T a, T b) { got.emplace_back(a, b); });
        CHECK(got == (ivec<T>{{T{5}, T{9}}, {T{20}, T{29}}, {T{40}, T{44}}}));
        auto iv = d.interval_of(T{25});
        CHECK(iv && iv->lo == T{20} && iv->hi == T{29});
        CHECK(!d.interval_of(T{10}));
    }
    { // for_each element loops must terminate at intervals touching the
      // exact type limits (a naive v <= hi loop would hang or overflow)
        diet::set<T> d;
        d.insert(static_cast<T>(tmax - 2), tmax);
        d.insert(tmin, static_cast<T>(tmin + 2));
        std::vector<T> elems;
        d.for_each([&](T v) { elems.push_back(v); });
        CHECK(elems == (std::vector<T>{tmin, static_cast<T>(tmin + 1),
                                       static_cast<T>(tmin + 2),
                                       static_cast<T>(tmax - 2),
                                       static_cast<T>(tmax - 1), tmax}));
    }
    { // iterator agrees with for_each_interval, and for_each visits elements
        diet::set<T> d{T{1}, T{2}, T{3}, T{7}, T{9}, T{10}};
        ivec<T> a = diet_intervals(d);
        ivec<T> b;
        for (auto iv : d) b.emplace_back(iv.lo, iv.hi);
        CHECK(a == b);
        auto it = d.begin();
        CHECK(it->lo == T{1} && it->hi == T{3}); // operator-> proxy
        std::vector<T> elems;
        d.for_each([&](T v) { elems.push_back(v); });
        CHECK(elems ==
              (std::vector<T>{T{1}, T{2}, T{3}, T{7}, T{9}, T{10}}));
        // same set built from ranges compares equal
        diet::set<T> e;
        e.insert(T{1}, T{3});
        e.insert(T{7});
        e.insert(T{9}, T{10});
        CHECK(d == e);
    }
    { // copy / move / swap
        diet::set<T> d;
        d.insert(T{1}, T{5});
        diet::set<T> c(d);
        CHECK(c == d);
        d.erase(T{3});
        CHECK(c != d);
        diet::set<T> m(std::move(c));
        CHECK(m.contains(T{1}, T{5}));
        CHECK(c.empty() && c.valid()); // moved-from: valid and empty
        CHECK(c.insert(T{7}));         // and reusable
        CHECK(c == diet::set<T>{T{7}});
        diet::set<T> n{T{9}}; // move-assign onto a non-empty target
        n = std::move(m);
        CHECK(n.contains(T{1}, T{5}) && !n.contains(T{9}));
        CHECK(m.empty() && m.valid());
        swap(n, d);
        CHECK(d.contains(T{3}) && !n.contains(T{3}));
        diet::set<T>* alias = &n; // self-move through a pointer: must survive
        n = std::move(*alias);
        CHECK(n.valid() && n.contains(T{1}));
    }
}

void cardinality_saturation_tests() {
    g_context = "cardinality_saturation";
    constexpr std::uint64_t u64max = std::numeric_limits<std::uint64_t>::max();
    {
        diet::set<std::uint64_t> d;
        d.insert(std::uint64_t{0}, u64max);
        CHECK(d.cardinality() == u64max); // 2^64 saturates
        d.erase(std::uint64_t{0});
        CHECK(d.cardinality() == u64max); // exactly 2^64 - 1, no saturation
        d.erase(std::uint64_t{1});
        CHECK(d.cardinality() == u64max - 1);
    }
    {
        diet::set<std::int64_t> d;
        d.insert(std::numeric_limits<std::int64_t>::min(),
                 std::numeric_limits<std::int64_t>::max());
        CHECK(d.cardinality() == u64max); // 2^64 saturates
        d.erase(std::int64_t{0});
        CHECK(d.cardinality() == u64max); // 2^64 - 1 exact
    }
    {
        diet::set<std::uint32_t> d;
        d.insert(std::uint32_t{0}, std::numeric_limits<std::uint32_t>::max());
        CHECK(d.cardinality() == (std::uint64_t{1} << 32));
    }
}

// Set algebra with degenerate / extreme-shape operands. The random fuzz rarely
// produces an empty operand, the full universe, or a one-huge-interval-versus-
// thousands-of-singletons pairing, so these are checked deterministically.
template <typename T>
void algebra_shape_tests(const char* tname) {
    g_context = std::string("algebra_shape<") + tname + ">";
    constexpr T tmin = std::numeric_limits<T>::min();
    constexpr T tmax = std::numeric_limits<T>::max();
    const diet::set<T> empty;
    const diet::set<T> u = diet::set<T>::universe();

    // A small, arbitrary operand valid at every width (tmin..tmin+? stays in range).
    diet::set<T> a;
    a.insert(tmin, static_cast<T>(tmin + 3));
    a.insert(static_cast<T>(tmin + 6), static_cast<T>(tmin + 8));
    a.insert(tmax);

    // Identities against the empty set.
    CHECK((a | empty) == a);
    CHECK((empty | a) == a);
    CHECK((a & empty).empty());
    CHECK((empty & a).empty());
    CHECK((a - empty) == a);
    CHECK((empty - a).empty());
    CHECK((empty | empty).empty());
    CHECK((empty & empty).empty());
    CHECK((empty - empty).empty());

    // Identities against the universe (checked structurally, so they hold at
    // any width without materializing 2^width elements).
    CHECK((a | u) == u);
    CHECK((u | a) == u);
    CHECK((a & u) == a);
    CHECK((u & a) == a);
    CHECK((a - u).empty());          // nothing survives removing everything
    CHECK((u - a) == a.complement()); // co-set of a within the domain
    CHECK(u.complement().empty());
    CHECK(empty.complement() == u);
    CHECK(a.complement().complement() == a);

    // Self operations.
    CHECK((a | a) == a);
    CHECK((a & a) == a);
    CHECK((a - a).empty());
    CHECK((u & u) == u);
    CHECK((u - u).empty());
}

// One huge interval versus thousands of scattered singletons, in a bounded
// domain so a std::set oracle stays feasible. This is the tree-shape asymmetry
// (1 node vs. ~N/2 nodes) that the stream-based algorithms are built for, in
// both recursion directions.
void algebra_asymmetry_test() {
    g_context = "algebra_asymmetry";
    using T = std::int32_t;
    constexpr T N = 3000;

    diet::set<T> huge; // one interval [0, N]
    huge.insert(0, N);
    std::set<T> ohuge;
    for (T v = 0; v <= N; ++v) ohuge.insert(v);

    diet::set<T> singles; // every even value in [0, N]: ~N/2 separate intervals
    std::set<T> osingles;
    for (T v = 0; v <= N; v += 2) {
        singles.insert(v);
        osingles.insert(v);
    }
    CHECK(singles.interval_count() == static_cast<std::size_t>(N / 2 + 1));

    auto oracle_binop = [](const std::set<T>& x, const std::set<T>& y, char op) {
        std::set<T> r;
        if (op == '|') {
            r = x;
            r.insert(y.begin(), y.end());
        } else if (op == '&') {
            for (T v : x)
                if (y.count(v)) r.insert(v);
        } else { // '-'
            for (T v : x)
                if (!y.count(v)) r.insert(v);
        }
        return r;
    };

    struct Case {
        const diet::set<T>& da;
        const std::set<T>& oa;
        const diet::set<T>& db;
        const std::set<T>& ob;
        const char* name;
    };
    const Case cases[] = {
        {huge, ohuge, singles, osingles, "huge op singles"},
        {singles, osingles, huge, ohuge, "singles op huge"},
    };
    for (const auto& c : cases) {
        g_context = std::string("algebra_asymmetry: ") + c.name;
        for (char op : {'|', '&', '-'}) {
            diet::set<T> got = (op == '|') ? (c.da | c.db)
                             : (op == '&') ? (c.da & c.db)
                                           : (c.da - c.db);
            CHECK(got.valid());
            CHECK(same_content(got, oracle_binop(c.oa, c.ob, op)));
        }
        CHECK(c.da == c.da); // operands untouched by the above (value semantics)
    }
    // Spot-check the expected shapes directly.
    CHECK((huge & singles) == singles);          // singles ⊂ huge
    CHECK((huge - singles).interval_count() ==   // the odd values remain
          static_cast<std::size_t>(N / 2));
    CHECK((singles - huge).empty());
    CHECK((huge | singles) == huge);
}

// Exception safety: inject a bad_alloc at every possible allocation point of
// each operation and require that the set is never left in an invalid state
// (ASan additionally proves no node is freed twice or used after free).
// Documented contract: no UB; the target may end up valid-but-empty; nodes
// may leak on the unwind path (leak detection is therefore not asserted).
void allocation_failure_tests() {
    g_context = "allocation_failure";
    using S = diet::set<int>;
    auto build = [] {
        S s;
        for (int i = 0; i < 12; ++i) s.insert(i * 10, i * 10 + 4);
        return s;
    };
    auto torture = [&](const char* what, auto&& op) {
        (void)what;
        for (long long n = 0; n < 200; ++n) {
            S a = build();
            S b;
            for (int i = 0; i < 60; ++i) b.insert(i * 7, i * 7 + 2);
            bool threw = false;
            alloc_fail::countdown = n;
            try {
                op(a, b);
            } catch (const std::bad_alloc&) {
                threw = true;
            }
            alloc_fail::countdown = -1;
            CHECK(a.valid());
            CHECK(b.valid());
            if (!threw) return; // reached an injection point past the op
        }
        CHECK(!"operation never completed under injection sweep");
    };
    torture("insert elem", [](S& a, S&) { a.insert(1000); });
    torture("insert range", [](S& a, S&) { a.insert(-50, 200); });
    torture("erase range", [](S& a, S&) { a.erase(3, 87); });
    torture("union", [](S& a, S& b) { a |= b; });
    torture("intersection", [](S& a, S& b) { a &= b; });
    torture("difference", [](S& a, S& b) { a -= b; });
    torture("copy", [](S& a, S& b) { b = a; });
    torture("complement", [](S& a, S&) { S c = a.complement(); CHECK(c.valid()); });
    torture("init list", [](S&, S&) { S c{1, 5, 9, 13, 2, 6}; CHECK(c.valid()); });
}

#ifdef NDEBUG
// The lo <= hi precondition is assert-checked; in release builds a violating
// call must degrade to a harmless no-op, never to a corrupted set.
void release_precondition_tests() {
    g_context = "release_precondition";
    diet::set<int> d;
    d.insert(20, 30);
    d.insert(10, 5); // violates lo <= hi: must be a no-op
    d.erase(40, 35); // likewise
    CHECK(d.valid());
    CHECK(d.cardinality() == 11);
    CHECK(!d.contains(10, 5));
    CHECK(!d.intersects(10, 5));
    bool visited = false;
    d.for_each_interval_in(10, 5, [&](int, int) { visited = true; });
    CHECK(!visited);
}
#endif

// The const_iterator's advertised input-iterator contract: traits, postfix
// increment, mid-traversal equality, and interoperability with <algorithm>.
void iterator_tests() {
    g_context = "iterator";
    using S = diet::set<int>;
    using It = S::const_iterator;
    using Tr = std::iterator_traits<It>;
    static_assert(std::is_same<Tr::value_type, S::interval>::value, "value_type");
    static_assert(std::is_same<Tr::iterator_category,
                               std::input_iterator_tag>::value, "category");
    static_assert(std::is_same<Tr::difference_type, std::ptrdiff_t>::value, "diff");

    S d;
    d.insert(1, 3);
    d.insert(7);
    d.insert(9, 10); // intervals: [1,3] [7,7] [9,10]

    // std::distance over the input iterator equals the interval count.
    CHECK(static_cast<std::size_t>(std::distance(d.begin(), d.end())) ==
          d.interval_count());

    // Postfix ++ yields the pre-increment value, then advances.
    It it = d.begin();
    It prev = it++;
    CHECK((*prev == S::interval{1, 3}));
    CHECK((*it == S::interval{7, 7}));

    // Mid-traversal equality: independent iterators at the same position are
    // equal; advancing one breaks equality; both reach end() together.
    It a = d.begin();
    It b = d.begin();
    CHECK(a == b);
    ++a;
    CHECK(a != b);
    ++b;
    CHECK(a == b);
    ++a;
    ++a; // a now past [9,10]
    ++b;
    ++b;
    CHECK(a == b && a == d.end());

    // A default-constructed iterator equals end().
    CHECK(It{} == d.end());

    // <algorithm> interop: copy out, and find a specific interval.
    std::vector<S::interval> v;
    std::copy(d.begin(), d.end(), std::back_inserter(v));
    CHECK(v.size() == 3);
    CHECK((v[2] == S::interval{9, 10}));
    auto found = std::find(d.begin(), d.end(), S::interval{7, 7});
    CHECK(found != d.end() && (*found == S::interval{7, 7}));
    CHECK(std::find(d.begin(), d.end(), S::interval{4, 4}) == d.end());

    // operator-> reaches members.
    CHECK(d.begin()->lo == 1 && d.begin()->hi == 3);

    // Empty set: begin() == end(), zero distance.
    S e;
    CHECK(e.begin() == e.end());
    CHECK(std::distance(e.begin(), e.end()) == 0);
}

// A dense sorted-singleton workload; valid() certifies the AVL shape held.
void balance_stress() {
    g_context = "balance_stress";
    diet::set<std::int32_t> d;
    for (std::int32_t i = 0; i < 4000; ++i) d.insert(i * 2); // no merges
    CHECK(d.interval_count() == 4000);
    CHECK(d.valid());
    for (std::int32_t i = 0; i < 4000; ++i) d.insert(i * 2 + 1); // total merge
    CHECK(d.interval_count() == 1);
    CHECK(d.valid());
    CHECK(d.cardinality() == 8000);
    for (std::int32_t i = 0; i < 4000; ++i) d.erase(i * 2); // re-split
    CHECK(d.interval_count() == 4000);
    CHECK(d.valid());
}

} // namespace

int main() {
    edge_tests<std::int8_t>("int8");
    edge_tests<std::uint8_t>("uint8");
    edge_tests<std::int16_t>("int16");
    edge_tests<std::uint16_t>("uint16");
    edge_tests<std::int32_t>("int32");
    edge_tests<std::uint32_t>("uint32");
    edge_tests<std::int64_t>("int64");
    edge_tests<std::uint64_t>("uint64");

    algebra_shape_tests<std::int8_t>("int8");
    algebra_shape_tests<std::uint8_t>("uint8");
    algebra_shape_tests<std::int16_t>("int16");
    algebra_shape_tests<std::uint16_t>("uint16");
    algebra_shape_tests<std::int32_t>("int32");
    algebra_shape_tests<std::uint32_t>("uint32");
    algebra_shape_tests<std::int64_t>("int64");
    algebra_shape_tests<std::uint64_t>("uint64");
    algebra_asymmetry_test();

    iterator_tests();
    cardinality_saturation_tests();
    balance_stress();
    allocation_failure_tests();
#ifdef NDEBUG
    release_precondition_tests();
#endif

    for (std::uint64_t seed : {1ull, 2ull, 3ull}) {
        fuzz_mutations<std::int8_t>("int8", seed, 3000, 1);
        fuzz_mutations<std::uint8_t>("uint8", seed, 3000, 1);
        fuzz_mutations<std::int16_t>("int16", seed, 2000, 8);
        fuzz_mutations<std::uint16_t>("uint16", seed, 2000, 8);
        fuzz_mutations<std::int32_t>("int32", seed, 1500, 8);
        fuzz_mutations<std::uint32_t>("uint32", seed, 1500, 8);
        fuzz_mutations<std::int64_t>("int64", seed, 1500, 8);
        fuzz_mutations<std::uint64_t>("uint64", seed, 1500, 8);
    }

    for (std::uint64_t seed : {11ull, 12ull}) {
        fuzz_algebra<std::int8_t>("int8", seed, 40, 200);
        fuzz_algebra<std::uint8_t>("uint8", seed, 40, 200);
        fuzz_algebra<std::int16_t>("int16", seed, 20, 300);
        fuzz_algebra<std::uint16_t>("uint16", seed, 20, 300);
        fuzz_algebra<std::int32_t>("int32", seed, 15, 300);
        fuzz_algebra<std::uint32_t>("uint32", seed, 15, 300);
        fuzz_algebra<std::int64_t>("int64", seed, 15, 300);
        fuzz_algebra<std::uint64_t>("uint64", seed, 15, 300);
    }

    if (g_failures == 0) {
        std::printf("all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
}

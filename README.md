# diet::set — a balanced DIET in C++

A header-only C++17 implementation of the **Discrete Interval Encoding Tree**
(DIET): an integer set stored as an AVL tree whose nodes hold *maximal runs*
`[lo, hi]` of consecutive values. Dense ("fat") sets cost one node per run
instead of one node per element, and every operation is bounded by the number
of **intervals**, never the number of elements — the right shape for key-range
queries over densely populated key spaces.

## Files

| File | Purpose |
|------|---------|
| `diet.hpp` | The implementation (single header, no dependencies). |
| `test_diet.cpp` | Oracle-based randomized tests + edge tests at type extremes. |
| `LICENSE.txt` | MIT license for the C++ code in this directory. |

## License

The C++ code in this directory (`diet.hpp`, `test_diet.cpp`) is copyright (c)
2026 Eden Networks Ltd., authored by Philip Lamb, and released under the MIT
license — see [`LICENSE.txt`](LICENSE.txt).

## References

- Martin Erwig, *Diets for Fat Sets*, JFP 8(6), 1998 — core DIET
  insert/delete/member.
- Oliver Friedmann, Martin Lange, *More on Balanced Diets*, JFP 21(2), 2011. -
  balanced diets and the stream-based union/intersection/difference algorithms
  implemented here. (Their `unionhelp2` pseudocode drops the result of its
  first-branch `insert` — a typo this port corrects.)
- Yamagata Yoriyuki's OCaml `ISet`/`AvlTree` (Camomile) — a reference
  implementation this port mirrors for the AVL primitives: bal, join
  ("make_tree"), concat, split_leftmost/rightmost.

## Usage

```cpp
#include "diet.hpp"

diet::set<std::uint32_t> s;
s.insert(5);                  // single element; returns true if it was absent
s.insert(100, 199);           // whole range [100, 199]
s.erase(150);                 // splits [100,199] into [100,149] [151,199]
s.erase(120, 129);            // range erase

s.contains(101);              // true
s.contains(100, 110);         // is the whole range present? true
s.intersects(140, 160);       // does any element fall in the range? true
s.interval_of(160);           // -> optional interval {151, 199}

for (auto [lo, hi] : s) { /* maximal intervals, ascending */ }
s.for_each_interval_in(110, 155, [](auto lo, auto hi) {
    /* stored runs clipped to the query range: [110,119] [130,149] [151,155] */
});

s.cardinality();              // element count (uint64_t, saturating)
s.interval_count();           // number of stored runs

auto u = s | other;           // union        (Friedmann & Lange, §3.3.3)
auto i = s & other;           // intersection (§3.3.1)
auto d = s - other;           // difference   (§3.3.2)
auto c = s.complement();      // within T's full domain
```

`T` may be any signed or unsigned integer type of 8–64 bits (`bool` excluded).
All arithmetic is overflow-safe at the exact type limits: `diet::set<uint8_t>`
happily holds `[0, 255]`, `diet::set<int64_t>` holds `INT64_MIN`, and a full
64-bit domain reports a (saturated) cardinality of `UINT64_MAX`.

## Complexities (I = number of stored intervals)

| Operation | Cost |
|-----------|------|
| `contains`, `interval_of`, `intersects`, `insert(x)`, `erase(x)` | O(log I) |
| `insert(lo,hi)`, `erase(lo,hi)` | O(log I + k), k = intervals absorbed/removed |
| `\|`, `&`, `-` | O(I log I) worst case (paper, Lemma 2), plus the O(I) copy inherent to value semantics |
| `cardinality`, `interval_count`, iteration, `complement` | O(I) |

## Preconditions and exception safety

Range arguments must satisfy `lo <= hi`; this is checked with `assert`, and a
violating call degrades to a harmless no-op (queries return false/nothing) in
release builds. `min_elt()`/`max_elt()`/`interval_of()` return `std::optional`
rather than asserting.

Only allocation can throw (`std::bad_alloc`). Single-element `insert`/`erase`
leave the set unchanged if it does. Range insert/erase and the set-algebra
operators guarantee no undefined behavior: the target is left in a valid
(possibly empty) state, though already-detached nodes may leak on that path.
Point queries and the interval visitors never allocate; the `begin()` iterator
(and `operator==` on sets, which uses it) holds a small heap traversal stack, so
those may allocate. The test suite sweeps a fail-at-Nth-allocation injector
across every mutating operation under AddressSanitizer to enforce this contract.

## Tests

```bash
clang++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
    -Wall -Wextra -Wpedantic test_diet.cpp -o test_diet
ASAN_OPTIONS=detect_leaks=0 ./test_diet
```

The allocation-failure tests leak intentionally (the documented contract permits
leaks on the throw-unwind path), so run with `ASAN_OPTIONS=detect_leaks=0` where
LeakSanitizer is on by default (Linux). macOS ASan does not run LeakSanitizer,
so a bare `./test_diet` also works there.

The suite mirrors every mutation into a `std::set<T>` oracle and compares the
full maximal-interval decomposition, the structural invariant (`valid()`:
AVL heights, balance factors, ascending independent intervals), cardinality,
min/max, and the range queries (`contains`, `intersects`, `for_each_interval_in`)
after (nearly) every operation, across all eight integer widths; set algebra is
fuzzed against oracle results — plus deterministic empty/universe/asymmetric
operand shapes — including operand immutability, aliasing (`a |= a`), algebraic
identities, exception safety, and the iterator contract.

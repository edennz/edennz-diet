// diet.hpp — Discrete Interval Encoding Tree (DIET) over an AVL tree.
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
// A diet::set<T> represents a set of integers of type T as a height-balanced
// binary search tree whose nodes hold maximal runs [lo, hi] of consecutive
// values ("independent intervals": disjoint and never adjacent). Densely
// populated ("fat") sets therefore need one node per run, not one per element,
// and every operation costs O(log I) or O(I) in the number of intervals I,
// never in the number of elements.
//
// References:
//   - M. Erwig, "Diets for Fat Sets", J. Functional Programming 8(6), 1998.
//     (the core DIET insert/delete/member algorithms)
//   - O. Friedmann, M. Lange, "More on Balanced Diets", JFP 21(2), 2011.
//     (balanced diets; stream-based union/intersection/difference)
//   - Yamagata Yoriyuki's OCaml ISet/AvlTree (Camomile library), a reference
//     implementation this port mirrors for the AVL primitives: bal, join
//     ("make_tree"), concat, split_leftmost/rightmost.
//
// T may be any signed or unsigned integer type of 8..64 bits (bool excluded).
// All interval arithmetic is overflow-safe up to the exact type limits, so
// e.g. set<uint8_t> can hold [0,255] and set<int64_t> can hold INT64_MIN.
//
// Exception safety: only allocation can throw (std::bad_alloc). Single-element
// insert/erase leave the set unchanged if it does. Range insert/erase and the
// set-algebra operators guarantee no undefined behavior: the target set is
// left in a valid (possibly empty) state, though already-detached nodes may
// leak. Point queries and the visitor-style traversals (for_each_interval,
// for_each_interval_in, for_each) never allocate. The iterator returned by
// begin() keeps a small heap-allocated traversal stack, so constructing or
// incrementing it, and operator== / operator!= on sets (which iterate), may
// allocate and throw std::bad_alloc.
//
// Requires C++17. Header-only; no dependencies beyond the standard library.

#ifndef DIET_HPP_INCLUDED
#define DIET_HPP_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace diet {

template <typename T>
class set {
    static_assert(std::is_integral_v<T>, "diet::set requires an integer type");
    static_assert(std::is_same_v<T, std::remove_cv_t<T>>,
                  "diet::set element type must not be const/volatile-qualified");
    static_assert(!std::is_same_v<T, bool>, "diet::set does not support bool");
    static_assert(sizeof(T) <= 8, "diet::set supports at most 64-bit integers");

public:
    using value_type = T;

    struct interval {
        T lo;
        T hi;
        friend bool operator==(const interval& a, const interval& b) {
            return a.lo == b.lo && a.hi == b.hi;
        }
        friend bool operator!=(const interval& a, const interval& b) {
            return !(a == b);
        }
    };

private:
    struct node {
        T lo;
        T hi;
        node* left;
        node* right;
        std::int8_t height; // AVL height; <= ~1.44*64+2 < 96 for any 64-bit domain
    };

    node* root_ = nullptr;

    static constexpr T t_min() { return std::numeric_limits<T>::min(); }
    static constexpr T t_max() { return std::numeric_limits<T>::max(); }

    // ---- overflow-safe successor/predecessor predicates -------------------
    // adj(a, b): a + 1 == b. gap(a, b): a + 1 < b (at least one value strictly
    // between a and b is missing). The a < b guard makes the +1 free of
    // overflow: when it is evaluated, a < b <= t_max() so a < t_max().
    static bool adj(T a, T b) { return a < b && static_cast<T>(a + 1) == b; }
    static bool gap(T a, T b) { return a < b && static_cast<T>(a + 1) != b; }

    // ---- node helpers ------------------------------------------------------
    static node* new_node(T lo, T hi) { return new node{lo, hi, nullptr, nullptr, 1}; }
    static void del_node(node* n) { delete n; } // n only; children must be detached

    static void free_tree(node* t) noexcept {
        if (!t) return;
        free_tree(t->left);
        free_tree(t->right);
        delete t;
    }

    static node* clone(const node* t) {
        if (!t) return nullptr;
        node* l = clone(t->left);
        node* r = nullptr;
        try {
            r = clone(t->right);
            return new node{t->lo, t->hi, l, r, t->height};
        } catch (...) {
            free_tree(l);
            free_tree(r);
            throw;
        }
    }

    static int h(const node* t) { return t ? t->height : 0; }

    // Relink n with children l, r and recompute its height.
    // Precondition: |h(l) - h(r)| <= 1.
    static node* mk(node* n, node* l, node* r) {
        n->left = l;
        n->right = r;
        n->height = static_cast<std::int8_t>(1 + std::max(h(l), h(r)));
        return n;
    }

    // Rebuild n over children l, r, restoring the AVL invariant with at most
    // a double rotation. Precondition: |h(l) - h(r)| <= 2.  (avlTree.ml: bal)
    static node* bal(node* n, node* l, node* r) {
        const int hl = h(l), hr = h(r);
        if (hl >= hr + 2) {
            node* ll = l->left;
            node* lr = l->right;
            if (h(ll) >= h(lr)) return mk(l, ll, mk(n, lr, r));
            node* a = mk(l, ll, lr->left);
            node* b = mk(n, lr->right, r);
            return mk(lr, a, b);
        }
        if (hr >= hl + 2) {
            node* rl = r->left;
            node* rr = r->right;
            if (h(rr) >= h(rl)) return mk(r, mk(n, l, rl), rr);
            node* a = mk(n, l, rl->left);
            node* b = mk(r, rl->right, rr);
            return mk(rl, a, b);
        }
        return mk(n, l, r);
    }

    // Insert detached node n as the minimum/maximum of t. (avlTree.ml: add_left/add_right)
    static node* add_min(node* n, node* t) {
        if (!t) return mk(n, nullptr, nullptr);
        node* l = add_min(n, t->left);
        return bal(t, l, t->right);
    }
    static node* add_max(node* n, node* t) {
        if (!t) return mk(n, nullptr, nullptr);
        node* r = add_max(n, t->right);
        return bal(t, t->left, r);
    }

    // Build a balanced tree from n intervals given in strictly ascending,
    // pairwise-independent order (the middle-root construction). O(n), and the
    // subtree sizes at every node differ by at most one, so the result is
    // AVL-valid with exact heights. Exception-safe: frees partial work on throw.
    static node* build_sorted(const interval* iv, std::size_t n) {
        if (n == 0) return nullptr;
        const std::size_t mid = n / 2;
        node* l = build_sorted(iv, mid);
        node* r = nullptr;
        try {
            r = build_sorted(iv + mid + 1, n - mid - 1);
            return mk(new_node(iv[mid].lo, iv[mid].hi), l, r);
        } catch (...) {
            free_tree(l);
            free_tree(r);
            throw;
        }
    }

    // Join two trees around detached separator node n, for arbitrary height
    // difference; O(|h(l) - h(r)|). (avlTree.ml: make_tree; the paper's l a⋈b r)
    static node* join(node* l, node* n, node* r) {
        if (!l) return add_min(n, r);
        if (!r) return add_max(n, l);
        if (h(l) > h(r) + 1) {
            node* nr = join(l->right, n, r);
            return bal(l, l->left, nr);
        }
        if (h(r) > h(l) + 1) {
            node* nl = join(l, n, r->left);
            return bal(r, nl, r->right);
        }
        return mk(n, l, r);
    }

    // Join two trees, all of a < all of b, no separator. (avlTree.ml: concat;
    // the paper's reroot l ⋈ r)
    static node* concat(node* a, node* b) {
        if (!a) return b;
        if (!b) return a;
        if (h(a) < h(b)) {
            node* bl = b->left;
            node* br = b->right;
            node* l = concat(a, bl);
            return join(l, b, br);
        }
        node* al = a->left;
        node* ar = a->right;
        node* r = concat(ar, b);
        return join(al, a, r);
    }

    static node* min_node(node* t) {
        while (t && t->left) t = t->left;
        return t;
    }
    static node* max_node(node* t) {
        while (t && t->right) t = t->right;
        return t;
    }
    static const node* min_node(const node* t) {
        while (t && t->left) t = t->left;
        return t;
    }
    static const node* max_node(const node* t) {
        while (t && t->right) t = t->right;
        return t;
    }

    // Detach the maximum/minimum node of t (returned in out, children null)
    // and return the rebalanced remainder. Precondition: t != nullptr.
    static node* split_max(node* t, node*& out) {
        if (!t->right) {
            out = t;
            node* l = t->left;
            t->left = nullptr;
            return l;
        }
        node* r = split_max(t->right, out);
        return bal(t, t->left, r);
    }
    static node* split_min(node* t, node*& out) {
        if (!t->left) {
            out = t;
            node* r = t->right;
            t->right = nullptr;
            return r;
        }
        node* l = split_min(t->left, out);
        return bal(t, l, t->right);
    }

    // ---- element insert / erase (Erwig; iSet.ml: add / remove) ------------
    static node* add_elem(node* t, T x, bool& changed) {
        if (!t) {
            changed = true;
            return new_node(x, x);
        }
        if (gap(x, t->lo)) { // x <= lo - 2: strictly left, no merge possible
            node* l = add_elem(t->left, x, changed);
            return bal(t, l, t->right);
        }
        if (gap(t->hi, x)) { // x >= hi + 2: strictly right
            node* r = add_elem(t->right, x, changed);
            return bal(t, t->left, r);
        }
        if (adj(x, t->lo)) { // x == lo - 1: extend downward, maybe absorb left max
            changed = true;
            t->lo = x;
            if (node* m = max_node(t->left); m && adj(m->hi, t->lo)) {
                node* d = nullptr;
                node* l = split_max(t->left, d);
                t->lo = d->lo;
                del_node(d);
                return bal(t, l, t->right);
            }
            return t;
        }
        if (adj(t->hi, x)) { // x == hi + 1: extend upward, maybe absorb right min
            changed = true;
            t->hi = x;
            if (node* m = min_node(t->right); m && adj(t->hi, m->lo)) {
                node* d = nullptr;
                node* r = split_min(t->right, d);
                t->hi = d->hi;
                del_node(d);
                return bal(t, t->left, r);
            }
            return t;
        }
        return t; // lo <= x <= hi: already present
    }

    static node* remove_elem(node* t, T x, bool& changed) {
        if (!t) return nullptr;
        if (x < t->lo) {
            node* l = remove_elem(t->left, x, changed);
            return bal(t, l, t->right);
        }
        if (x > t->hi) {
            node* r = remove_elem(t->right, x, changed);
            return bal(t, t->left, r);
        }
        changed = true;
        if (t->lo == t->hi) { // whole interval vanishes
            node* l = t->left;
            node* r = t->right;
            del_node(t);
            return concat(l, r);
        }
        if (x == t->lo) { // shrink from below; lo < hi so +1 is safe
            t->lo = static_cast<T>(t->lo + 1);
            return t;
        }
        if (x == t->hi) { // shrink from above
            t->hi = static_cast<T>(t->hi - 1);
            return t;
        }
        // Interior removal: keep [x+1, hi] here, push [lo, x-1] down as the
        // new maximum of the left subtree (lo < x < hi, so both +-1 are safe).
        node* piece = new_node(t->lo, static_cast<T>(x - 1));
        t->lo = static_cast<T>(x + 1);
        node* l = add_max(piece, t->left);
        return bal(t, l, t->right);
    }

    // ---- three-way split for range operations ------------------------------
    // Destructively splits t at x: lt holds all elements < x, ge all >= x.
    // An interval straddling x is cut in two (one extra node allocated).
    struct split_result {
        node* lt;
        node* ge;
    };

    static split_result split_at(node* t, T x) {
        if (!t) return {nullptr, nullptr};
        if (x <= t->lo) {
            node* l = t->left;
            node* r = t->right;
            t->left = nullptr;
            split_result s = split_at(l, x);
            return {s.lt, join(s.ge, t, r)};
        }
        if (x > t->hi) {
            node* l = t->left;
            node* r = t->right;
            t->right = nullptr;
            split_result s = split_at(r, x);
            return {join(l, t, s.lt), s.ge};
        }
        // t->lo < x <= t->hi: cut [lo,hi] into [lo, x-1] | [x, hi]
        node* l = t->left;
        node* r = t->right;
        t->left = t->right = nullptr;
        node* low = new_node(t->lo, static_cast<T>(x - 1));
        t->lo = x;
        return {add_max(low, l), add_min(t, r)};
    }

    // ---- merging attach helpers (keep intervals maximal) -------------------
    // Append [a, b] as the new maximum of l, fusing with l's current maximum
    // when adjacent. Precondition: max(l) < a and [a,b] does not overlap l.
    static node* append_max(node* l, T a, T b) {
        if (node* m = max_node(l); m && adj(m->hi, a)) {
            m->hi = b; // in-place extension: shape and heights unchanged
            return l;
        }
        return add_max(new_node(a, b), l);
    }

    // join that first fuses m with an adjacent neighbor on either flank.
    static node* join_merge(node* l, node* m, node* r) {
        if (node* p = max_node(l); p && adj(p->hi, m->lo)) {
            node* d = nullptr;
            l = split_max(l, d);
            m->lo = d->lo;
            del_node(d);
        }
        if (node* p = min_node(r); p && adj(m->hi, p->lo)) {
            node* d = nullptr;
            r = split_min(r, d);
            m->hi = d->hi;
            del_node(d);
        }
        return join(l, m, r);
    }

    // concat that fuses across the seam when max(a) is adjacent to min(b).
    static node* concat_merge(node* a, node* b) {
        if (!a || !b) return a ? a : b;
        node* pb = min_node(b);
        if (node* pa = max_node(a); adj(pa->hi, pb->lo)) {
            node* d = nullptr;
            a = split_max(a, d);
            pb->lo = d->lo; // in-place extension of b's minimum
            del_node(d);
        }
        return concat(a, b);
    }

    // ---- stream: lazy ascending decomposition (paper, Section 3.1) --------
    // Owns and consumes a tree, yielding its intervals in ascending order.
    // Implemented as the unwound left spine instead of the paper's rotations;
    // right subtrees stay intact (with valid heights) for cheap reassembly.
    class stream {
        std::vector<node*> spine_; // back() = smallest pending interval

        // The spine is an in-order traversal stack, so it never grows beyond
        // the tree height; an AVL tree over at most 2^63 intervals has height
        // < 96. Reserving more than that up front makes every later
        // push_back nonthrowing.
        static constexpr std::size_t spine_reserve = 128;

        void descend(node* n) {
            while (n) {
                spine_.push_back(n);
                node* l = n->left;
                n->left = nullptr;
                n = l;
            }
        }

    public:
        explicit stream(node* t) {
            try {
                spine_.reserve(spine_reserve);
            } catch (...) {
                free_tree(t); // reserve failed before any node changed hands
                throw;
            }
            descend(t);
        }
        stream(const stream&) = delete;
        stream& operator=(const stream&) = delete;
        ~stream() {
            for (node* n : spine_) free_tree(n); // left is null; frees n + right
        }

        bool empty() const { return spine_.empty(); }
        T front_lo() const { return spine_.back()->lo; }
        T front_hi() const { return spine_.back()->hi; }

        void pop() {
            node* n = spine_.back();
            spine_.pop_back();
            node* r = n->right;
            n->right = nullptr;
            del_node(n);
            descend(r);
        }

        // Extend the front interval downward in place (the paper's "push the
        // merged interval back onto the stream", allocation-free: the front
        // node's queued right subtree is unaffected because only lo moves).
        // Precondition: v <= front_lo(), and v is greater than every value
        // already consumed from the stream.
        void front_set_lo(T v) { spine_.back()->lo = v; }

        // Rebuild the remainder into an AVL tree, reusing the intact right
        // subtrees. Entries from front (index 0) hold the largest intervals.
        node* to_tree() {
            node* acc = nullptr;
            for (node* n : spine_) { // index 0 first: largest downward
                node* r = n->right;
                n->right = nullptr;
                acc = add_min(n, concat(r, acc));
            }
            spine_.clear();
            return acc;
        }
    };

    // ---- intersection (paper, Section 3.3.1) -------------------------------
    // Consumes t and advances s; returns the tree of [[t]] ∩ [[s]].
    static node* inter_rec(node* t, stream& s) {
        if (!t) return nullptr;
        if (s.empty()) {
            free_tree(t);
            return nullptr;
        }
        node* l = t->left;
        node* r = t->right;
        t->left = t->right = nullptr;
        node* l2 = nullptr;
        if (s.front_lo() < t->lo) {
            l2 = inter_rec(l, s);
        } else {
            free_tree(l); // stream starts at/after t->lo: nothing of l survives
        }
        return inter_help(l2, t, r, s);
    }

    // l ∪ (([x,y],⊥,r) ∩ s) where m carries [x,y]; l is already final and
    // strictly below the overlap region.
    static node* inter_help(node* l, node* m, node* r, stream& s) {
        for (;;) {
            if (s.empty()) {
                del_node(m);
                free_tree(r);
                return l;
            }
            const T xs = s.front_lo();
            const T ys = s.front_hi();
            if (ys < m->lo) { // stream interval entirely below [x,y]
                s.pop();
                continue;
            }
            if (m->hi < xs) { // stream starts beyond [x,y]: [x,y] contributes nothing
                node* r2 = inter_rec(r, s);
                del_node(m);
                return concat(l, r2);
            }
            if (!gap(ys, m->hi)) { // ys >= hi - 1: overlap reaches the top of [x,y]
                m->lo = std::max(m->lo, xs);
                m->hi = std::min(m->hi, ys);
                node* r2 = inter_rec(r, s); // front may still overlap r; not popped
                return join(l, m, r2);
            }
            // ys <= hi - 2: emit the overlap piece, keep processing [ys+2.., y].
            l = append_max(l, std::max(m->lo, xs), ys);
            m->lo = static_cast<T>(ys + 1); // ys < hi - 1 < t_max()
            s.pop();
        }
    }

    // ---- difference (paper, Section 3.3.2) ---------------------------------
    // Consumes t and advances s; returns the tree of [[t]] \ [[s]].
    static node* diff_rec(node* t, stream& s) {
        if (!t || s.empty()) return t;
        node* l = t->left;
        node* r = t->right;
        t->left = t->right = nullptr;
        node* l2 = (s.front_lo() >= t->lo) ? l : diff_rec(l, s);
        return diff_help(l2, t, r, s);
    }

    static node* diff_help(node* l, node* m, node* r, stream& s) {
        for (;;) {
            if (s.empty()) return join(l, m, r);
            const T xs = s.front_lo();
            const T ys = s.front_hi();
            if (ys < m->lo) { // stream interval entirely below: irrelevant
                s.pop();
                continue;
            }
            if (m->hi < xs) { // stream beyond [x,y]: [x,y] survives whole
                node* r2 = diff_rec(r, s);
                return join(l, m, r2);
            }
            if (m->lo < xs) { // keep the uncovered prefix [lo, xs-1]
                l = append_max(l, m->lo, static_cast<T>(xs - 1));
                m->lo = xs; // front not popped: it may still cover more
                continue;
            }
            if (ys < m->hi) { // covered prefix removed; continue with [ys+1, hi]
                m->lo = static_cast<T>(ys + 1);
                s.pop();
                continue;
            }
            // [x,y] fully covered by the front interval.
            del_node(m);
            node* r2 = diff_rec(r, s); // front may still cover parts of r
            return concat(l, r2);
        }
    }

    // ---- union (paper, Section 3.3.3) --------------------------------------
    // Consumes t and advances s. The limit (has_eps, eps) caps how far a
    // merged interval may grow inside a left subtree: anything reaching eps
    // is pushed back onto the stream for the parent to merge with its own
    // interval. has_eps == false means "no limit" (the paper's ⊤).
    static node* union_rec(node* t, stream& s, bool has_eps, T eps) {
        if (!t || s.empty()) return t;
        node* l = t->left;
        node* r = t->right;
        t->left = t->right = nullptr;
        node* l2 = l;
        if (s.front_lo() < t->lo) {
            // t->lo > front_lo >= t_min(), so t->lo - 1 is safe.
            l2 = union_rec(l, s, true, static_cast<T>(t->lo - 1));
        }
        return union_help(l2, t, r, s, has_eps, eps);
    }

    static node* union_help(node* l, node* m, node* r, stream& s,
                            bool has_eps, T eps) {
        for (;;) {
            if (s.empty()) return join_merge(l, m, r);
            const T xs = s.front_lo();
            const T ys = s.front_hi();
            if (gap(ys, m->lo)) { // front entirely below and not adjacent
                s.pop();
                l = append_max(l, xs, ys);
                // (The paper's pseudocode drops this insert result — a known
                // typo; the inserted interval must be kept.)
                continue;
            }
            if (gap(m->hi, xs)) { // front entirely above and not adjacent
                node* r2 = union_rec(r, s, has_eps, eps);
                return join_merge(l, m, r2);
            }
            if (ys <= m->hi) { // front absorbed into [x,y]
                if (xs < m->lo) m->lo = xs;
                s.pop();
                continue;
            }
            // Front overlaps/adjoins [x,y] and extends beyond hi: the front
            // interval becomes the merged [min(lo,xs), ys] in place, which
            // swallows [x,y].
            const T i = std::min(m->lo, xs);
            del_node(m);
            s.front_set_lo(i);
            if (has_eps && ys >= eps) {
                // The merged interval reaches the limit: leave [i, ys] on the
                // stream for the parent. r lies strictly between hi and eps,
                // so it is entirely swallowed by [i, ys].
                free_tree(r);
                return l;
            }
            node* r2 = union_rec(r, s, has_eps, eps);
            return concat_merge(l, r2);
        }
    }

    // Drivers taking ownership of both roots.
    static node* union_roots(node* t, node* s_tree) {
        stream s(s_tree);
        node* t2 = union_rec(t, s, false, T{});
        return concat_merge(t2, s.to_tree());
    }
    static node* inter_roots(node* t, node* s_tree) {
        stream s(s_tree);
        return inter_rec(t, s); // stream destructor frees any remainder
    }
    static node* diff_roots(node* t, node* s_tree) {
        stream s(s_tree);
        return diff_rec(t, s);
    }

    // ---- misc recursive helpers -------------------------------------------
    template <typename F>
    static void walk(const node* t, F& f) {
        if (!t) return;
        walk(t->left, f);
        f(t->lo, t->hi);
        walk(t->right, f);
    }

    template <typename F>
    static void walk_in(const node* t, T lo, T hi, F& f) {
        if (!t) return;
        if (lo < t->lo) walk_in(t->left, lo, hi, f);
        if (t->lo <= hi && lo <= t->hi) // overlap with the query range
            f(std::max(t->lo, lo), std::min(t->hi, hi));
        if (hi > t->hi) walk_in(t->right, lo, hi, f);
    }

    static std::size_t count_nodes(const node* t) {
        return t ? 1 + count_nodes(t->left) + count_nodes(t->right) : 0;
    }

    static void card_rec(const node* t, std::uint64_t& total, bool& sat) {
        if (!t || sat) return;
        using U = std::make_unsigned_t<T>;
        const std::uint64_t span = static_cast<std::uint64_t>(
            static_cast<U>(static_cast<U>(t->hi) - static_cast<U>(t->lo)));
        if (span >= std::numeric_limits<std::uint64_t>::max() - total) {
            sat = true; // only reachable for a (nearly) full 64-bit domain
            return;
        }
        total += span + 1;
        card_rec(t->left, total, sat);
        card_rec(t->right, total, sat);
    }

    static int check_shape(const node* t, bool& ok) {
        if (!t) return 0;
        const int hl = check_shape(t->left, ok);
        const int hr = check_shape(t->right, ok);
        if (t->height != 1 + std::max(hl, hr)) ok = false;
        if (hl - hr > 1 || hr - hl > 1) ok = false;
        return 1 + std::max(hl, hr);
    }

    static void check_order(const node* t, const node*& prev, bool& ok) {
        if (!t) return;
        check_order(t->left, prev, ok);
        if (t->lo > t->hi) ok = false;
        if (prev && !gap(prev->hi, t->lo)) ok = false; // must leave a hole
        prev = t;
        check_order(t->right, prev, ok);
    }

public:
    // ---- construction / assignment ----------------------------------------
    set() = default;

    set(std::initializer_list<T> xs) {
        try {
            for (T x : xs) insert(x);
        } catch (...) {
            free_tree(root_); // the destructor will not run for a throwing ctor
            throw;
        }
    }

    set(const set& other) : root_(clone(other.root_)) {}

    set(set&& other) noexcept : root_(other.root_) { other.root_ = nullptr; }

    set& operator=(const set& other) {
        if (this != &other) {
            node* copy = clone(other.root_);
            free_tree(root_);
            root_ = copy;
        }
        return *this;
    }

    set& operator=(set&& other) noexcept {
        if (this != &other) {
            free_tree(root_);
            root_ = other.root_;
            other.root_ = nullptr;
        }
        return *this;
    }

    ~set() { free_tree(root_); }

    void swap(set& other) noexcept { std::swap(root_, other.root_); }
    friend void swap(set& a, set& b) noexcept { a.swap(b); }

    // The full domain [min, max] of T as a single interval.
    static set universe() {
        set s;
        s.root_ = new_node(t_min(), t_max());
        return s;
    }

    // ---- basic queries -----------------------------------------------------
    bool empty() const { return root_ == nullptr; }

    void clear() {
        free_tree(root_);
        root_ = nullptr;
    }

    bool contains(T x) const {
        const node* t = root_;
        while (t) {
            if (x < t->lo)
                t = t->left;
            else if (x > t->hi)
                t = t->right;
            else
                return true;
        }
        return false;
    }

    // Is every value in [lo, hi] present? Requires lo <= hi.
    bool contains(T lo, T hi) const {
        assert(lo <= hi);
        if (lo > hi) return false;
        const node* t = root_;
        while (t) {
            if (hi < t->lo)
                t = t->left;
            else if (lo > t->hi)
                t = t->right;
            else // some overlap; intervals are maximal, so containment is local
                return t->lo <= lo && hi <= t->hi;
        }
        return false;
    }

    // Is any value in [lo, hi] present? Requires lo <= hi.
    bool intersects(T lo, T hi) const {
        assert(lo <= hi);
        if (lo > hi) return false;
        const node* t = root_;
        while (t) {
            if (hi < t->lo)
                t = t->left;
            else if (lo > t->hi)
                t = t->right;
            else
                return true;
        }
        return false;
    }

    // The maximal interval containing x, if x is present.
    std::optional<interval> interval_of(T x) const {
        const node* t = root_;
        while (t) {
            if (x < t->lo)
                t = t->left;
            else if (x > t->hi)
                t = t->right;
            else
                return interval{t->lo, t->hi};
        }
        return std::nullopt;
    }

    std::optional<T> min_elt() const {
        const node* m = min_node(root_);
        return m ? std::optional<T>(m->lo) : std::nullopt;
    }
    std::optional<T> max_elt() const {
        const node* m = max_node(root_);
        return m ? std::optional<T>(m->hi) : std::nullopt;
    }

    // Number of stored intervals (tree nodes). O(intervals).
    std::size_t interval_count() const { return count_nodes(root_); }

    // Number of elements, saturating at UINT64_MAX (a full 64-bit domain
    // holds 2^64 elements, which does not fit). O(intervals).
    std::uint64_t cardinality() const {
        std::uint64_t total = 0;
        bool sat = false;
        card_rec(root_, total, sat);
        return sat ? std::numeric_limits<std::uint64_t>::max() : total;
    }

    // ---- modification ------------------------------------------------------
    // Insert one element; returns true if the set changed.
    bool insert(T x) {
        bool changed = false;
        root_ = add_elem(root_, x, changed);
        return changed;
    }

    // Insert every value in [lo, hi]; requires lo <= hi (checked by assert;
    // a violating call is a no-op in release builds). (iSet.ml: add_range)
    void insert(T lo, T hi) {
        assert(lo <= hi);
        if (lo > hi) return;
        // Detach the tree before any node is freed: if an allocation below
        // throws, the set is left empty (never pointing at freed nodes).
        node* t = root_;
        root_ = nullptr;
        split_result a = split_at(t, lo); // a.lt < lo <= a.ge
        node* right = nullptr;
        if (hi == t_max()) {
            free_tree(a.ge); // everything >= lo is swallowed by [lo, hi]
        } else {
            split_result b = split_at(a.ge, static_cast<T>(hi + 1));
            free_tree(b.lt); // swallowed: values in [lo, hi]
            right = b.ge;
        }
        T nlo = lo, nhi = hi;
        if (node* m = max_node(a.lt); m && adj(m->hi, nlo)) {
            node* d = nullptr;
            a.lt = split_max(a.lt, d);
            nlo = d->lo;
            del_node(d);
        }
        if (node* m = min_node(right); m && adj(nhi, m->lo)) {
            node* d = nullptr;
            right = split_min(right, d);
            nhi = d->hi;
            del_node(d);
        }
        root_ = join(a.lt, new_node(nlo, nhi), right);
    }

    // Erase one element; returns true if the set changed.
    bool erase(T x) {
        bool changed = false;
        root_ = remove_elem(root_, x, changed);
        return changed;
    }

    // Erase every value in [lo, hi]; requires lo <= hi (checked by assert;
    // a violating call is a no-op in release builds). (iSet.ml: remove_range)
    void erase(T lo, T hi) {
        assert(lo <= hi);
        if (lo > hi) return;
        node* t = root_;
        root_ = nullptr; // as in insert(lo, hi): never leave root_ dangling
        split_result a = split_at(t, lo);
        node* right = nullptr;
        if (hi == t_max()) {
            free_tree(a.ge);
        } else {
            split_result b = split_at(a.ge, static_cast<T>(hi + 1));
            free_tree(b.lt);
            right = b.ge;
        }
        root_ = concat(a.lt, right); // the removed span keeps them independent
    }

    // ---- set algebra (Friedmann & Lange 2011, Section 3.3) -----------------
    set& operator|=(const set& other) {
        if (other.empty() || root_ == other.root_) return *this;
        if (empty()) return *this = other;
        set cp(other);
        // Recurse over the taller tree; stream the shorter one.
        node* t;
        node* s;
        if (h(root_) >= h(cp.root_)) {
            t = root_;
            s = cp.root_;
        } else {
            t = cp.root_;
            s = root_;
        }
        root_ = nullptr;
        cp.root_ = nullptr;
        root_ = union_roots(t, s);
        return *this;
    }

    set& operator&=(const set& other) {
        if (root_ == other.root_) return *this;
        if (empty() || other.empty()) {
            clear();
            return *this;
        }
        set cp(other);
        node* t;
        node* s;
        if (h(root_) >= h(cp.root_)) {
            t = root_;
            s = cp.root_;
        } else {
            t = cp.root_;
            s = root_;
        }
        root_ = nullptr;
        cp.root_ = nullptr;
        root_ = inter_roots(t, s);
        return *this;
    }

    set& operator-=(const set& other) {
        if (root_ == other.root_) {
            clear();
            return *this;
        }
        if (empty() || other.empty()) return *this;
        set cp(other);
        node* t = root_;
        root_ = nullptr;
        node* s = cp.root_;
        cp.root_ = nullptr;
        root_ = diff_roots(t, s);
        return *this;
    }

    friend set operator|(set a, const set& b) { return a |= b, std::move(a); }
    friend set operator&(set a, const set& b) { return a &= b, std::move(a); }
    friend set operator-(set a, const set& b) { return a -= b, std::move(a); }

    // Complement within the full domain of T. O(I): the up-to-I+1 gap
    // intervals are collected in ascending order and turned into a balanced
    // tree by a single bottom-up build.
    set complement() const {
        std::vector<interval> gaps;
        gaps.reserve(interval_count() + 1);
        T cursor = t_min();
        bool open = true; // is [cursor, ...] still to be emitted?
        for_each_interval([&](T a, T b) {
            if (cursor < a) gaps.push_back({cursor, static_cast<T>(a - 1)});
            if (b == t_max()) {
                open = false;
            } else {
                cursor = static_cast<T>(b + 1);
            }
        });
        if (open) gaps.push_back({cursor, t_max()});
        set out;
        out.root_ = build_sorted(gaps.data(), gaps.size());
        return out;
    }

    // ---- iteration ---------------------------------------------------------
    // Visit every maximal interval in ascending order: f(lo, hi).
    template <typename F>
    void for_each_interval(F&& f) const {
        walk(root_, f);
    }

    // Visit the portions of stored intervals that intersect [lo, hi],
    // clipped to the query range: f(max(lo, ilo), min(hi, ihi)).
    template <typename F>
    void for_each_interval_in(T lo, T hi, F&& f) const {
        assert(lo <= hi);
        if (lo > hi) return;
        walk_in(root_, lo, hi, f);
    }

    // Visit every element in ascending order. Beware: this enumerates
    // elements, so it is O(cardinality), not O(intervals).
    template <typename F>
    void for_each(F&& f) const {
        for_each_interval([&f](T a, T b) {
            for (T v = a;; ++v) {
                f(v);
                if (v == b) break; // avoids overflowing past b == t_max()
            }
        });
    }

    // Input iterator over maximal intervals (ascending). Dereferences by value.
    class const_iterator {
        std::vector<const node*> stack_;

        void descend(const node* n) {
            while (n) {
                stack_.push_back(n);
                n = n->left;
            }
        }

        // operator-> must return something with a real interval to point at.
        struct arrow_proxy {
            interval v;
            const interval* operator->() const { return &v; }
        };

    public:
        using value_type = interval;
        using reference = interval;
        using pointer = arrow_proxy;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        const_iterator() = default;
        explicit const_iterator(const node* root) { descend(root); }

        interval operator*() const {
            const node* n = stack_.back();
            return {n->lo, n->hi};
        }

        arrow_proxy operator->() const { return {**this}; }

        const_iterator& operator++() {
            const node* n = stack_.back();
            stack_.pop_back();
            descend(n->right);
            return *this;
        }

        const_iterator operator++(int) {
            const_iterator tmp(*this);
            ++*this;
            return tmp;
        }

        friend bool operator==(const const_iterator& a, const const_iterator& b) {
            if (a.stack_.empty() || b.stack_.empty())
                return a.stack_.empty() == b.stack_.empty();
            return a.stack_.back() == b.stack_.back();
        }
        friend bool operator!=(const const_iterator& a, const const_iterator& b) {
            return !(a == b);
        }
    };

    const_iterator begin() const { return const_iterator(root_); }
    const_iterator end() const { return const_iterator(); }

    // ---- comparison --------------------------------------------------------
    // Set equality: the maximal-interval decomposition is canonical, so two
    // sets are equal iff their interval sequences are equal.
    friend bool operator==(const set& a, const set& b) {
        const_iterator ia = a.begin(), ib = b.begin();
        const const_iterator ea = a.end(), eb = b.end();
        while (ia != ea && ib != eb) {
            if (*ia != *ib) return false;
            ++ia;
            ++ib;
        }
        return ia == ea && ib == eb;
    }
    friend bool operator!=(const set& a, const set& b) { return !(a == b); }

    // ---- diagnostics -------------------------------------------------------
    // Verify the full representation invariant: correct AVL heights, balance
    // factors in [-1, 1], lo <= hi per node, and strictly ascending intervals
    // separated by at least one absent value. Intended for tests/debugging.
    bool valid() const {
        bool ok = true;
        check_shape(root_, ok);
        const node* prev = nullptr;
        check_order(root_, prev, ok);
        return ok;
    }
};

} // namespace diet

#endif // DIET_HPP_INCLUDED

import assert from "node:assert/strict";
import test from "node:test";
import React from "react";
import type { DependencyList } from "react";
import { VirtualizedList } from "../src/components/VirtualizedList";

type ListProps = Parameters<typeof VirtualizedList<string>>[0];
type Hook = { value?: unknown; deps?: DependencyList; cleanup?: () => void };

// Commit refs and effects around the real component without a browser or renderer.
function withList(initial: Partial<ListProps>, check: (list: ReturnType<typeof createList>) => void) {
  const hooks: Hook[] = [];
  let cursor = 0;
  let dirty = false;
  let effects: Array<() => void> = [];
  const takeHook = () => {
    const index = cursor++;
    return hooks[index] ?? (hooks[index] = {});
  };
  const changed = (previous: DependencyList | undefined, next: DependencyList | undefined) =>
    !previous || !next || previous.length !== next.length || next.some((value, index) => !Object.is(value, previous[index]));
  const originalHooks = {
    useRef: React.useRef, useState: React.useState, useMemo: React.useMemo,
    useCallback: React.useCallback, useEffect: React.useEffect,
  };
  Object.assign(React, {
    useRef: (initialValue: unknown) => {
      const hook = takeHook();
      if (!("value" in hook)) hook.value = { current: initialValue };
      return hook.value;
    },
    useState: (initialValue: unknown) => {
      const hook = takeHook();
      if (!("value" in hook)) hook.value = typeof initialValue === "function" ? initialValue() : initialValue;
      return [hook.value, (value: unknown) => {
        const next = typeof value === "function" ? value(hook.value) : value;
        if (!Object.is(next, hook.value)) { hook.value = next; dirty = true; }
      }];
    },
    useMemo: (compute: () => unknown, deps: DependencyList) => {
      const hook = takeHook();
      if (changed(hook.deps, deps)) { hook.value = compute(); hook.deps = deps; }
      return hook.value;
    },
    useCallback: (callback: unknown, deps: DependencyList) => {
      const hook = takeHook();
      if (changed(hook.deps, deps)) { hook.value = callback; hook.deps = deps; }
      return hook.value;
    },
    useEffect: (effect: () => void | (() => void), deps?: DependencyList) => {
      const hook = takeHook();
      if (changed(hook.deps, deps)) {
        hook.deps = deps;
        effects.push(() => { hook.cleanup?.(); hook.cleanup = effect() || undefined; });
      }
    },
  });
  const originalObserver = Object.getOwnPropertyDescriptor(globalThis, "ResizeObserver");
  const observers: FakeResizeObserver[] = [];
  class FakeResizeObserver {
    observed = new Set<Element>();
    constructor(readonly callback: ResizeObserverCallback) { observers.push(this); }
    observe(node: Element) { this.observed.add(node); }
    unobserve(node: Element) { this.observed.delete(node); }
    disconnect() { this.observed.clear(); }
  }
  Object.defineProperty(globalThis, "ResizeObserver", { configurable: true, value: FakeResizeObserver });

  function createList() {
    const scrolls: ScrollToOptions[] = [];
    const node = {
      clientHeight: 200,
      scrollTop: 0,
      scrollTo: (options: ScrollToOptions) => { scrolls.push(options); node.scrollTop = options.top ?? 0; },
    };
    const itemNodes = new Map<number, HTMLElement>();
    let focused: number[] = [];
    let props: ListProps = {
      itemCount: 8, estimatedItemHeight: 100, overscan: 8,
      getItem: (index) => `item-${index}`,
      renderItem: (item, index, selected) => { if (selected) focused.push(index); return item; },
      ...initial,
    };
    let element: ReturnType<typeof VirtualizedList<string>>;
    const render = (updates: Partial<ListProps> = {}) => {
      props = { ...props, ...updates };
      let commits = 0;
      do {
        assert.ok(++commits < 20, "The component should settle after committing its effects.");
        dirty = false;
        cursor = 0;
        effects = [];
        focused = [];
        element = VirtualizedList(props);
        (element as unknown as { ref: { current: unknown } }).ref.current = node;
        for (const row of element.props.children.props.children) {
          const index = row.props["data-virtual-index"] as number;
          const itemNode = itemNodes.get(index) ?? ({ dataset: { virtualIndex: String(index) } } as unknown as HTMLElement);
          itemNodes.set(index, itemNode);
          row.ref(itemNode);
        }
        for (const effect of effects) effect();
      } while (dirty);
    };
    render();
    return {
      render, scrolls, node,
      focused: () => focused,
      manualScroll: (top: number) => {
        node.scrollTop = top;
        element.props.onScroll({ currentTarget: node });
        render();
      },
      measure: (index: number, height: number) => {
        const target = itemNodes.get(index);
        assert.ok(target, "The measured card must be mounted.");
        const observer = observers.find((value) => value.observed.has(target));
        assert.ok(observer, "The component must observe the measured card.");
        observer.callback([{
          target, borderBoxSize: [{ blockSize: height }], contentRect: { height },
        } as unknown as ResizeObserverEntry], observer as unknown as ResizeObserver);
        render();
      },
    };
  }

  try { check(createList()); }
  finally {
    for (const hook of hooks) hook.cleanup?.();
    Object.assign(React, originalHooks);
    if (originalObserver) Object.defineProperty(globalThis, "ResizeObserver", originalObserver);
    else Reflect.deleteProperty(globalThis, "ResizeObserver");
  }
}

test("initial navigation and legacy highlighting occur once despite manual scroll and new measurements", () => {
  withList({ scrollToIndex: 4 }, (list) => {
    assert.deepEqual(list.scrolls, [{ top: 392, behavior: "smooth" }]);
    assert.deepEqual(list.focused(), [4]);
    list.manualScroll(100);
    list.measure(0, 180);
    list.measure(1, 140);
    assert.equal(list.scrolls.length, 1);
    assert.equal(list.node.scrollTop, 100);
  });
});

test("changing focusedIndex updates highlighting without scrolling", () => {
  withList({ scrollToIndex: 2, focusedIndex: 2 }, (list) => {
    list.manualScroll(50);
    list.render({ focusedIndex: 5 });
    assert.deepEqual(list.focused(), [5]);
    assert.equal(list.scrolls.length, 1);
    assert.equal(list.node.scrollTop, 50);
    list.render({ focusedIndex: null });
    assert.deepEqual(list.focused(), []);
    assert.equal(list.scrolls.length, 1);
  });
});

test("a new request ID navigates again to the same target exactly once", () => {
  withList({ scrollToIndex: 3 }, (list) => {
    list.manualScroll(20);
    list.render({ scrollRequestId: 1 });
    assert.deepEqual(list.scrolls, [{ top: 292, behavior: "smooth" }, { top: 292, behavior: "smooth" }]);
    list.measure(0, 150);
    list.render({ scrollRequestId: 1 });
    assert.equal(list.scrolls.length, 2);
  });
});

test("changing the navigation target scrolls once even when measurements change", () => {
  withList({ scrollToIndex: 1 }, (list) => {
    list.render({ scrollToIndex: 5 });
    assert.deepEqual(list.scrolls.map((call) => call.top), [92, 492]);
    list.measure(0, 160);
    list.render({ itemCount: 10 });
    assert.equal(list.scrolls.length, 2);
  });
});

test("clearing the target rearms navigation to the same index and request ID", () => {
  withList({ scrollToIndex: 2, scrollRequestId: 7 }, (list) => {
    list.render({ scrollToIndex: null });
    list.manualScroll(25);
    list.render({ scrollToIndex: 2 });
    assert.deepEqual(list.scrolls.map((call) => call.top), [192, 192]);
    list.measure(0, 125);
    assert.equal(list.scrolls.length, 2);
  });
});

test("an initially unavailable target scrolls when the list contains it", () => {
  withList({ itemCount: 2, scrollToIndex: 4 }, (list) => {
    assert.equal(list.scrolls.length, 0);
    list.measure(0, 150);
    list.render({ itemCount: 5 });
    assert.deepEqual(list.scrolls, [{ top: 442, behavior: "smooth" }]);
    list.measure(1, 125);
    assert.equal(list.scrolls.length, 1);
  });
});

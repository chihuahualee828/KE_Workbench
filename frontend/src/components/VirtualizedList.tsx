"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

type VirtualizedListProps<T> = {
  itemCount: number;
  estimatedItemHeight: number;
  className?: string;
  overscan?: number;
  scrollToIndex?: number | null;
  getItem: (index: number) => T | undefined;
  onVisibleRangeChange?: (start: number, end: number) => void;
  renderItem: (item: T, index: number, focused: boolean) => ReactNode;
  renderPlaceholder?: (index: number) => ReactNode;
};

/**
 * Small dependency-free virtualizer for the app's independently scrollable
 * card lists.  It measures cards after they mount, so long generations do not
 * need a fixed card height.
 */
export function VirtualizedList<T>({
  itemCount,
  estimatedItemHeight,
  className = "",
  overscan = 3,
  scrollToIndex = null,
  getItem,
  onVisibleRangeChange,
  renderItem,
  renderPlaceholder,
}: VirtualizedListProps<T>) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const itemNodes = useRef(new Map<number, HTMLElement>());
  const observer = useRef<ResizeObserver | null>(null);
  const heights = useRef(new Map<number, number>());
  const [heightVersion, setHeightVersion] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);

  const offsets = useMemo(() => {
    const values = new Array<number>(itemCount + 1);
    values[0] = 0;
    for (let index = 0; index < itemCount; index += 1) {
      values[index + 1] = values[index] + (heights.current.get(index) ?? estimatedItemHeight);
    }
    return values;
  }, [estimatedItemHeight, heightVersion, itemCount]);

  const findIndex = useCallback((position: number) => {
    let lower = 0;
    let upper = Math.max(0, itemCount - 1);
    while (lower < upper) {
      const middle = Math.ceil((lower + upper) / 2);
      if (offsets[middle] <= position) lower = middle;
      else upper = middle - 1;
    }
    return lower;
  }, [itemCount, offsets]);

  const start = itemCount ? Math.max(0, findIndex(scrollTop) - overscan) : 0;
  const end = itemCount ? Math.min(itemCount - 1, findIndex(scrollTop + viewportHeight) + overscan) : -1;

  useEffect(() => {
    onVisibleRangeChange?.(start, end);
  }, [end, onVisibleRangeChange, start]);

  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    const updateViewport = () => setViewportHeight(node.clientHeight);
    updateViewport();
    const resizeObserver = new ResizeObserver(updateViewport);
    resizeObserver.observe(node);
    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      let changed = false;
      for (const entry of entries) {
        const index = Number((entry.target as HTMLElement).dataset.virtualIndex);
        const nextHeight = Math.ceil(entry.borderBoxSize?.[0]?.blockSize ?? entry.contentRect.height);
        if (!Number.isFinite(index) || nextHeight <= 0 || heights.current.get(index) === nextHeight) continue;
        heights.current.set(index, nextHeight);
        changed = true;
      }
      if (changed) setHeightVersion((current) => current + 1);
    });
    observer.current = resizeObserver;
    itemNodes.current.forEach((node) => resizeObserver.observe(node));
    return () => {
      resizeObserver.disconnect();
      observer.current = null;
    };
  }, []);

  useEffect(() => {
    if (scrollToIndex === null || scrollToIndex < 0 || scrollToIndex >= itemCount) return;
    scrollRef.current?.scrollTo({ top: Math.max(0, offsets[scrollToIndex] - 8), behavior: "smooth" });
  }, [itemCount, offsets, scrollToIndex]);

  const setItemNode = useCallback((index: number, node: HTMLElement | null) => {
    const previous = itemNodes.current.get(index);
    if (previous && previous !== node) observer.current?.unobserve(previous);
    if (!node) {
      itemNodes.current.delete(index);
      return;
    }
    itemNodes.current.set(index, node);
    observer.current?.observe(node);
  }, [scrollToIndex]);

  return (
    <div
      ref={scrollRef}
      className={className}
      onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
    >
      <div className="relative" style={{ height: offsets[itemCount] ?? 0 }}>
        {Array.from({ length: Math.max(0, end - start + 1) }, (_, position) => start + position).map((index) => {
          const item = getItem(index);
          return (
            <div
              key={index}
              data-virtual-index={index}
              ref={(node) => setItemNode(index, node)}
              className="absolute left-0 right-0 pb-3"
              style={{ top: offsets[index] }}
              tabIndex={item && index === scrollToIndex ? -1 : undefined}
            >
              {item ? renderItem(item, index, index === scrollToIndex) : renderPlaceholder?.(index) ?? <div className="h-24 animate-pulse rounded-md border border-line bg-panel/60" />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

import { useLayoutEffect } from "react";

export function useTableLayout() {
  // Every client shares card selection, bidding controls and analysis drawers.
  useLayoutEffect(() => {
    document.documentElement.classList.add("table-app");
    return () => document.documentElement.classList.remove("table-app");
  }, []);
  return true;
}

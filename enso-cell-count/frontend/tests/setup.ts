// Test environment setup.
//
// * Loads jest-dom for ergonomic DOM matchers (toBeInTheDocument, etc.).
// * Re-enables React's `act()` wrapping warnings in vitest.

import "@testing-library/jest-dom";

// React's `act` helper detects the environment via this global.
declare const globalThis: {
  IS_REACT_ACT_ENVIRONMENT?: boolean;
} & typeof global;

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

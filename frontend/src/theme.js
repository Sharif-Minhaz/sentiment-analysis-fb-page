import { ref, readonly } from 'vue'

const STORAGE_KEY = 'theme'
// Fall back to the operating system's preference rather than hardcoding light.
// The CSS dark block is guarded by :not([data-theme="light"]), so stamping
// "light" unconditionally would suppress the OS setting for every first-time
// visitor. An explicit choice, once made, is remembered and always wins.
const systemPrefersDark = () =>
  typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches

const theme = ref(
  localStorage.getItem(STORAGE_KEY) || (systemPrefersDark() ? 'dark' : 'light'),
)

// Charts bake their colors in at construction time, so they cannot pick up a
// CSS variable change on their own. Components register here and get called
// when the theme flips.
const listeners = new Set()

function apply() {
  document.documentElement.dataset.theme = theme.value
}

export function useTheme() {
  return {
    theme: readonly(theme),
    apply,
    toggle() {
      theme.value = theme.value === 'dark' ? 'light' : 'dark'
      localStorage.setItem(STORAGE_KEY, theme.value)
      apply()
      listeners.forEach((fn) => fn())
    },
    onThemeChange(fn) {
      listeners.add(fn)
      return () => listeners.delete(fn)
    },
  }
}

/**
 * Read a CSS custom property, so JS never hardcodes a palette value.
 *
 * Only for Chart.js, which paints to a canvas and needs a real color string.
 * Anything rendered as HTML should use `var(--name)` in its style instead:
 * the browser resolves that live, so it follows a theme toggle for free.
 */
export const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim()

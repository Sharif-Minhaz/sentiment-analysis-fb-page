/** Shared formatting helpers, so components don't each reinvent them. */

export const formatDay = (iso) =>
  iso ? new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : ''

export const percentOf = (n, total) => (total ? `${(n / total) * 100}%` : '0%')

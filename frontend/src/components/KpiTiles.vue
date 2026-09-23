<script setup>
import { computed } from 'vue'

const props = defineProps({
  totals: { type: Object, default: () => ({}) },
  percentages: { type: Object, default: () => ({}) },
})

// A single headline number is not a chart -- it's a stat tile. Four of them
// answer "how is the page doing" faster than any plot could.
//
// Colors are var() references, not cssVar() lookups: the browser resolves a
// var() live, so the swatch follows a theme toggle. A resolved hex would be
// cached by this computed (the theme isn't one of its reactive inputs) and
// stay stuck on the old theme's color.
const tiles = computed(() => [
  { label: 'Total',    value: props.totals.total    || 0, pct: null, color: null },
  { label: 'Positive', value: props.totals.positive || 0, pct: props.percentages.positive ?? 0, color: 'var(--positive)' },
  { label: 'Neutral',  value: props.totals.neutral  || 0, pct: props.percentages.neutral  ?? 0, color: 'var(--neutral)'  },
  { label: 'Negative', value: props.totals.negative || 0, pct: props.percentages.negative ?? 0, color: 'var(--negative)' },
])
</script>

<template>
  <section class="grid grid-cols-2 md:grid-cols-4 gap-3">
    <div v-for="tile in tiles" :key="tile.label" class="card p-4">
      <div class="flex items-center gap-2 mb-1">
        <span v-if="tile.color" class="w-2.5 h-2.5 rounded-sm" :style="{ background: tile.color }" />
        <span class="text-xs uppercase tracking-wide" :style="{ color: 'var(--muted)' }">
          {{ tile.label }}
        </span>
      </div>
      <div class="text-3xl font-semibold num">{{ tile.value.toLocaleString() }}</div>
      <div v-if="tile.pct !== null" class="text-sm num" :style="{ color: 'var(--ink-2)' }">
        {{ tile.pct }}% of all
      </div>
    </div>
  </section>
</template>

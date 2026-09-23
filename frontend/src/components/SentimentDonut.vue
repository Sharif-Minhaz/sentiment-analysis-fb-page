<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import Chart from 'chart.js/auto'
import { cssVar, useTheme } from '../theme'

const props = defineProps({
  totals: { type: Object, default: () => ({}) },
  percentages: { type: Object, default: () => ({}) },
})

const canvas = ref(null)
let chart = null
const { onThemeChange } = useTheme()

// Direct labels carrying the exact values. A donut shows proportion at a
// glance, but comparing two close arcs by eye is unreliable -- these make the
// comparison exact, and double as the accessible table view.
const legend = computed(() =>
  ['positive', 'neutral', 'negative'].map((key) => ({
    key,
    label: key[0].toUpperCase() + key.slice(1),
    value: props.totals[key] || 0,
    pct: props.percentages[key] ?? 0,
    // var(), not cssVar(): resolved live by the browser, so it follows a
    // theme toggle. Only the canvas needs resolved hex values.
    color: `var(--${key})`,
  })),
)

function draw() {
  chart?.destroy()
  if (!canvas.value) return

  const surface = cssVar('--surface')
  chart = new Chart(canvas.value, {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Neutral', 'Negative'],
      datasets: [{
        data: [props.totals.positive || 0, props.totals.neutral || 0, props.totals.negative || 0],
        backgroundColor: [cssVar('--positive'), cssVar('--neutral'), cssVar('--negative')],
        borderColor: surface,   // 2px surface gap between segments
        borderWidth: 2,
      }],
    },
    options: {
      cutout: '62%',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },   // rendered as direct labels below instead
        tooltip: {
          backgroundColor: surface,
          titleColor: cssVar('--ink'),
          bodyColor: cssVar('--ink-2'),
          borderColor: cssVar('--hairline'),
          borderWidth: 1,
          padding: 10,
          boxWidth: 8,
          boxHeight: 8,
        },
      },
    },
  })
}

onMounted(draw)
watch(() => props.totals, draw, { deep: true })
const stop = onThemeChange(draw)
onBeforeUnmount(() => { stop(); chart?.destroy() })
</script>

<template>
  <div class="card p-4">
    <h2 class="text-sm font-medium mb-1">Overall split</h2>
    <p class="text-xs mb-3" :style="{ color: 'var(--muted)' }">
      Share of all classified comments
    </p>

    <div class="relative" style="height: 220px">
      <canvas ref="canvas" />
    </div>

    <ul class="mt-3 space-y-1">
      <li v-for="row in legend" :key="row.key" class="flex items-center gap-2 text-sm">
        <span class="w-2.5 h-2.5 rounded-sm shrink-0" :style="{ background: row.color }" />
        <span :style="{ color: 'var(--ink-2)' }">{{ row.label }}</span>
        <span class="ml-auto num">{{ row.value.toLocaleString() }}</span>
        <span class="num w-12 text-right" :style="{ color: 'var(--muted)' }">{{ row.pct }}%</span>
      </li>
    </ul>
  </div>
</template>

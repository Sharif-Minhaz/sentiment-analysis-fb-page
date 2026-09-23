<script setup>
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'
import Chart from 'chart.js/auto'
import { cssVar, useTheme } from '../theme'
import { formatDay } from '../format'

const props = defineProps({ timeline: { type: Array, default: () => [] } })

const canvas = ref(null)
let chart = null
const { onThemeChange } = useTheme()

function draw() {
  chart?.destroy()
  if (!canvas.value) return

  const surface = cssVar('--surface')
  const muted = cssVar('--muted')

  // The API returns flat (day, sentiment, n) rows; Chart.js wants one dataset
  // per sentiment with a value for every day. Pivot here.
  const days = [...new Set(props.timeline.map((r) => r.day))].sort()
  const series = (name) =>
    days.map((d) => props.timeline.find((r) => r.day === d && r.sentiment === name)?.n || 0)

  chart = new Chart(canvas.value, {
    type: 'bar',
    data: {
      labels: days.map(formatDay),
      datasets: [
        { label: 'Negative', data: series('negative'), backgroundColor: cssVar('--negative') },
        { label: 'Neutral',  data: series('neutral'),  backgroundColor: cssVar('--neutral')  },
        {
          label: 'Positive',
          data: series('positive'),
          backgroundColor: cssVar('--positive'),
          // Rounded ends only on the top of the stack, anchored to the baseline
          borderRadius: { topLeft: 4, topRight: 4 },
        },
      ].map((d) => ({
        ...d,
        borderColor: surface,          // 2px surface gap between stacked segments
        borderWidth: { top: 2 },
        borderSkipped: false,
        // Without a cap, a single day of data renders as one bar stretched
        // across the entire plot, which reads as a filled area rather than a
        // bar. Keeps narrow date ranges legible.
        maxBarThickness: 48,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },   // crosshair-style hover
      plugins: {
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
        legend: {
          position: 'bottom',
          align: 'start',
          labels: {
            color: cssVar('--ink-2'),
            boxWidth: 8, boxHeight: 8,
            usePointStyle: true, pointStyle: 'rectRounded',
            padding: 14,
          },
        },
      },
      scales: {
        x: {
          stacked: true,
          grid: { display: false },
          border: { color: cssVar('--axis') },
          ticks: { color: muted, maxRotation: 0, autoSkipPadding: 12 },
        },
        y: {
          stacked: true,
          beginAtZero: true,
          grid: { color: cssVar('--grid'), drawTicks: false },   // recessive grid
          border: { display: false },
          ticks: { color: muted, precision: 0, padding: 8 },
        },
      },
    },
  })
}

onMounted(draw)
watch(() => props.timeline, draw, { deep: true })
const stop = onThemeChange(draw)
onBeforeUnmount(() => { stop(); chart?.destroy() })
</script>

<template>
  <div class="card p-4">
    <h2 class="text-sm font-medium mb-1">Over time</h2>
    <p class="text-xs mb-3" :style="{ color: 'var(--muted)' }">
      Comments per day by sentiment
    </p>
    <div class="relative" style="height: 300px">
      <canvas ref="canvas" />
    </div>
  </div>
</template>

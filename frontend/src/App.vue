<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import * as api from './api'
import { useTheme } from './theme'

import KpiTiles from './components/KpiTiles.vue'
import SentimentDonut from './components/SentimentDonut.vue'
import TimelineChart from './components/TimelineChart.vue'
import TryModel from './components/TryModel.vue'
import NeedsAttention from './components/NeedsAttention.vue'
import PostBreakdown from './components/PostBreakdown.vue'
import ModelInsights from './components/ModelInsights.vue'

const { theme, toggle } = useTheme()

const analytics = reactive({
  totals: {}, percentages: {}, timeline: [], per_post: [], needs_attention: [],
})
const model = reactive({})
const loading = ref(false)
const error = ref('')
const lastRun = ref('')

const accuracy = computed(() =>
  model.test_metrics ? (model.test_metrics.accuracy * 100).toFixed(1) : null,
)

async function loadAnalytics() {
  Object.assign(analytics, await api.getAnalytics())
}

/** The "View latest expression" button: fetch -> store -> classify -> redraw. */
async function refresh() {
  loading.value = true
  error.value = ''
  try {
    const data = await api.refresh()
    Object.assign(analytics, data.analytics)
    lastRun.value =
      `Fetched ${data.comments_seen} comments across ${data.posts_seen} posts · ` +
      `${data.new_comments} new · ${data.newly_classified} newly classified`
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  try {
    Object.assign(model, await api.getModel())
    await loadAnalytics()
  } catch (e) {
    error.value = e.message
  }
})
</script>

<template>
  <div class="min-h-screen">
    <div class="max-w-6xl mx-auto px-4 py-6 space-y-4">

      <header class="flex flex-wrap items-center gap-3">
        <div class="mr-auto">
          <h1 class="text-xl font-semibold">Page Sentiment</h1>
          <p class="text-sm" :style="{ color: 'var(--ink-2)' }">
            Facebook comment analysis &middot; TF-IDF + Logistic Regression
          </p>
        </div>

        <span
          v-if="accuracy"
          class="text-xs px-2 py-1 rounded-full border num"
          :style="{ borderColor: 'var(--hairline)', color: 'var(--ink-2)' }"
        >model {{ accuracy }}% acc</span>

        <button
          class="text-sm px-3 py-2 rounded-lg border"
          :style="{ borderColor: 'var(--hairline)' }"
          @click="toggle"
        >
          {{ theme === 'dark' ? 'Light' : 'Dark' }}
        </button>

        <button
          class="text-sm font-medium px-4 py-2 rounded-lg text-white disabled:opacity-50"
          :style="{ background: 'var(--positive)' }"
          :disabled="loading"
          @click="refresh"
        >
          {{ loading ? 'Fetching…' : 'View latest expression' }}
        </button>
      </header>

      <p
        v-if="error"
        class="text-sm px-3 py-2 rounded-lg card"
        :style="{ color: 'var(--negative)' }"
      >{{ error }}</p>

      <p v-if="lastRun" class="text-sm" :style="{ color: 'var(--ink-2)' }">{{ lastRun }}</p>

      <!-- Not connected: say so plainly rather than showing an empty dashboard
           that looks like a page with no comments. -->
      <section v-if="model.trained && model.configured === false" class="card p-4">
        <h2 class="text-sm font-medium mb-1">Facebook is not connected</h2>
        <p class="text-sm mb-2" :style="{ color: 'var(--ink-2)' }">
          There is nothing to show until the app can read your Page.
        </p>
        <pre class="text-xs p-3 rounded-lg overflow-x-auto" :style="{ background: 'var(--plane)', color: 'var(--ink-2)' }">cd backend
uv run python get_token.py</pre>
      </section>

      <!-- Connected but nothing fetched yet. -->
      <section
        v-else-if="model.configured && !analytics.totals.total"
        class="card p-4"
      >
        <h2 class="text-sm font-medium mb-1">No comments yet</h2>
        <p class="text-sm" :style="{ color: 'var(--ink-2)' }">
          Click <strong>View latest expression</strong> to pull the newest comments
          from your Page.
        </p>
      </section>

      <template v-if="analytics.totals.total">
      <KpiTiles :totals="analytics.totals" :percentages="analytics.percentages" />

      <section class="grid md:grid-cols-5 gap-3">
        <SentimentDonut
          class="md:col-span-2"
          :totals="analytics.totals"
          :percentages="analytics.percentages"
        />
        <TimelineChart class="md:col-span-3" :timeline="analytics.timeline" />
      </section>

      <NeedsAttention :comments="analytics.needs_attention" />
      <PostBreakdown :posts="analytics.per_post" />
      </template>

      <!-- These two work without any Facebook data, so they always show. -->
      <TryModel />
      <ModelInsights :model="model" />

    </div>
  </div>
</template>

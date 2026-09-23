<script setup>
import { ref } from 'vue'
import { classify } from '../api'

const text = ref('')
const result = ref(null)
const busy = ref(false)
const error = ref('')

async function run() {
  if (!text.value.trim()) return
  busy.value = true
  error.value = ''
  try {
    result.value = await classify(text.value)
  } catch (e) {
    error.value = e.message
    result.value = null
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <section class="card p-4">
    <h2 class="text-sm font-medium mb-1">Try the model</h2>
    <p class="text-xs mb-3" :style="{ color: 'var(--muted)' }">
      Type a sentence and see what the classifier makes of it
    </p>

    <div class="flex flex-wrap gap-2">
      <input
        v-model="text"
        placeholder="e.g. the delivery was really fast, thanks"
        class="flex-1 min-w-64 px-3 py-2 rounded-lg border text-sm"
        :style="{ background: 'var(--plane)', borderColor: 'var(--hairline)', color: 'var(--ink)' }"
        @keyup.enter="run"
      >
      <button
        class="text-sm px-4 py-2 rounded-lg border disabled:opacity-50"
        :style="{ borderColor: 'var(--hairline)' }"
        :disabled="busy"
        @click="run"
      >
        {{ busy ? 'Working…' : 'Classify' }}
      </button>
    </div>

    <p v-if="error" class="mt-3 text-sm" :style="{ color: 'var(--negative)' }">{{ error }}</p>

    <div v-if="result" class="mt-3 flex flex-wrap items-center gap-3 text-sm">
      <span class="w-2.5 h-2.5 rounded-sm" :style="{ background: `var(--${result.sentiment})` }" />
      <span class="font-medium capitalize">{{ result.sentiment }}</span>
      <span class="num" :style="{ color: 'var(--ink-2)' }">
        P(positive) = {{ result.score.toFixed(3) }}
      </span>
      <span class="num" :style="{ color: 'var(--muted)' }">
        confidence {{ (result.confidence * 100).toFixed(0) }}%
      </span>
      <span v-if="result.reason" class="text-xs" :style="{ color: 'var(--muted)' }">
        ({{ result.reason }})
      </span>
    </div>
  </section>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { formatDay } from '../format'

const props = defineProps({ comments: { type: Array, default: () => [] } })

const PER_PAGE = 7

const page = ref(1)
const pageCount = computed(() => Math.max(1, Math.ceil(props.comments.length / PER_PAGE)))

// Slice client-side. The server already caps this list at 50 -- it is a
// worklist, not an archive -- so paging in the browser avoids a round trip
// per page and keeps the interaction instant.
const start = computed(() => (page.value - 1) * PER_PAGE)
const visible = computed(() => props.comments.slice(start.value, start.value + PER_PAGE))

// A refresh can shrink the list. Without this, you could be stranded on page 4
// of a list that now has two pages, looking at an empty panel.
watch(() => props.comments, () => { page.value = 1 })

const go = (delta) => {
  page.value = Math.min(pageCount.value, Math.max(1, page.value + delta))
}
</script>

<template>
  <section v-if="comments.length" class="card p-4">
    <div class="flex flex-wrap items-baseline gap-x-3 mb-1">
      <h2 class="text-sm font-medium">Needs attention</h2>
      <span class="text-xs num ml-auto" :style="{ color: 'var(--muted)' }">
        {{ start + 1 }}–{{ Math.min(start + visible.length, comments.length) }}
        of {{ comments.length }}
      </span>
    </div>
    <p class="text-xs mb-3" :style="{ color: 'var(--muted)' }">
      Most negative comments first - these are the ones worth replying to
    </p>

    <ul class="divide-y divide-[color:var(--hairline)]">
      <li
        v-for="comment in visible"
        :key="comment.id"
        class="py-2 flex gap-3 items-start text-sm"
      >
        <span class="w-1 self-stretch rounded-full shrink-0" :style="{ background: 'var(--negative)' }" />
        <div class="min-w-0 flex-1">
          <p class="truncate">{{ comment.message }}</p>
          <p class="text-xs" :style="{ color: 'var(--muted)' }">
            {{ comment.author || 'Unknown' }} &middot; {{ formatDay(comment.created_time) }}
          </p>
        </div>
        <span class="num text-xs shrink-0" :style="{ color: 'var(--muted)' }">
          {{ comment.score.toFixed(2) }}
        </span>
      </li>
    </ul>

    <nav
      v-if="pageCount > 1"
      class="flex items-center gap-2 mt-3 pt-3 border-t"
      :style="{ borderColor: 'var(--hairline)' }"
    >
      <button
        class="text-sm px-3 py-1.5 rounded-lg border disabled:opacity-40 disabled:cursor-not-allowed"
        :style="{ borderColor: 'var(--hairline)' }"
        :disabled="page === 1"
        @click="go(-1)"
      >
        Previous
      </button>

      <!-- Page numbers, so you can jump rather than click through -->
      <div class="flex gap-1 mx-auto">
        <button
          v-for="n in pageCount"
          :key="n"
          class="w-8 h-8 text-sm rounded-lg num"
          :style="n === page
            ? { background: 'var(--negative)', color: '#fff' }
            : { color: 'var(--ink-2)' }"
          @click="page = n"
        >{{ n }}</button>
      </div>

      <button
        class="text-sm px-3 py-1.5 rounded-lg border disabled:opacity-40 disabled:cursor-not-allowed"
        :style="{ borderColor: 'var(--hairline)' }"
        :disabled="page === pageCount"
        @click="go(1)"
      >
        Next
      </button>
    </nav>
  </section>
</template>

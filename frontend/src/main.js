import { createApp } from 'vue'
import App from './App.vue'
import './style.css'
import { useTheme } from './theme'

useTheme().apply()   // set data-theme before first paint, to avoid a flash
createApp(App).mount('#app')

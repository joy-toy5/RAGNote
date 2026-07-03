import { defineStore } from 'pinia';

export const useThemeStore = defineStore('theme', {
  state: () => ({
    currentTheme: localStorage.getItem('theme') || 'light',
    themes: {
      light: {
        name: '浅色·温润',
        sidebar: '#EDE4D3',
        bg: '#F7EEDD',
        surface: '#EDE4D3',
        card: '#FFFFFF',
        text: '#000000',
        textLight: '#2C2C2C',
        textLighter: '#5C5C5C',
        textLightest: '#8C8C8C',
        primary: '#FF7F50',
        border: '#C4BCAB',
        borderLight: '#E0D8C7',
        divider: '#D8CFBE',
        shadow: 'rgba(0, 0, 0, 0.10)',
      },
      dark: {
        name: '深色·沉静',
        sidebar: '#0F1328',
        bg: '#151931',
        surface: '#252841',
        card: '#3D3F5B',
        text: '#A096A2',
        textLight: '#847A86',
        textLighter: '#6B6072',
        textLightest: '#555066',
        primary: '#E7D1BB',
        border: '#3D3F5B',
        borderLight: '#2E3150',
        divider: '#2A2D4A',
        shadow: 'rgba(0, 0, 0, 0.50)',
      },
    },
  }),

  getters: {
    getCurrentTheme: (state) => state.currentTheme,
    getThemeConfig: (state) => state.themes[state.currentTheme],
    getAllThemes: (state) =>
      Object.keys(state.themes).map((key) => ({
        id: key,
        name: state.themes[key].name,
        primaryColor: state.themes[key].primary,
        bgColor: state.themes[key].bg,
      })),
  },

  actions: {
    setTheme(themeName) {
      if (this.themes[themeName]) {
        this.currentTheme = themeName;
        localStorage.setItem('theme', themeName);
        this.applyTheme();
      }
    },

    applyTheme() {
      const t = this.themes[this.currentTheme];
      const root = document.documentElement;
      root.style.setProperty('--color-sidebar', t.sidebar);
      root.style.setProperty('--color-bg', t.bg);
      root.style.setProperty('--color-surface', t.surface);
      root.style.setProperty('--color-card', t.card);
      root.style.setProperty('--color-text', t.text);
      root.style.setProperty('--color-text-light', t.textLight);
      root.style.setProperty('--color-text-lighter', t.textLighter);
      root.style.setProperty('--color-text-lightest', t.textLightest);
      root.style.setProperty('--color-primary', t.primary);
      root.style.setProperty('--color-border', t.border);
      root.style.setProperty('--color-border-light', t.borderLight);
      root.style.setProperty('--color-divider', t.divider);
      root.style.setProperty('--color-shadow', t.shadow);
    },

    initTheme() {
      this.applyTheme();
    },
  },
});
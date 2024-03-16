import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import  './routerBefore'

import '@/styles/font-family.css'

import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';

Vue.use(ElementUI);

//自定义样式
import '@/styles/index.scss'

import clipboard from "clipboard";
Vue.prototype.clipboard = clipboard;

import dataV from '@jiaminghi/data-view'
Vue.use(dataV)

import * as filters from './filters';
Object.keys(filters).forEach(key => {
    Vue.filter(key, filters[key])
});

import {getApiBaseUrl,getImageBaseUrl} from '@/utils/app-base'
import LOG from './utils/self-log'
Vue.prototype.API_BASE_URL = getApiBaseUrl();
Vue.prototype.IMAGE_BASE_URL = getImageBaseUrl();
Vue.prototype.LOG = LOG;

Vue.config.productionTip = false;
new Vue({
  router,
  store,
  created(){
    window.ISLOG = false;
  },
  render: h => h(App)
}).$mount('#app')

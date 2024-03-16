import router from './router'
import NProgress from 'nprogress'
import 'nprogress/nprogress.css'
import store from './store'


router.beforeEach(async (to, from, next) => {
    NProgress.start();
    next()
    NProgress.done()
});

router.afterEach((to, from) => {
    NProgress.done()
    if (to.meta && to.meta.pageNav) {
        store.dispatch('app/setPageName', to.name)
    }
});

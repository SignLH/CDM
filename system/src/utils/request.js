import axios from 'axios'
import {
    getApiBaseUrl
} from '@/utils/app-base'

const service = axios.create({
    withCredentials: false, // send cookies when cross-domain requests
    timeout: 600000, // request timeout
    headers: {
        'Content-Type': 'application/json;charset=UTF-8'
    },
});

// request interceptor
service.interceptors.request.use(
    config => {
        if (!config.baseURL) {
            config.baseURL = getApiBaseUrl();
        }
        let loginToken = sessionStorage.getItem('LoginToken')
        if (loginToken != null) {
            config.headers['token'] = loginToken
        }
        return config
    },
    error => {
        return Promise.reject(error)
    }
);

// response interceptor
service.interceptors.response.use(
    response => {
        return response.data
    },
    error => {
        return Promise.reject(error)
    }
);

export default service
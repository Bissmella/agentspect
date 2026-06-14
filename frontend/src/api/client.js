import axios from 'axios';

const client = axios.create();

client.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const detail = error.response?.data?.detail || error.message;
    const status = error.response?.status || 0;
    return Promise.reject({ status, detail });
  },
);

export default client;

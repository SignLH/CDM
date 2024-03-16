const LOG = {
  info(log) {
    if (process.env.NODE_ENV === 'development' || window.ISLOG) {
      console.info(...log)
    }
  },
  warn(log) {
    if (process.env.NODE_ENV === 'development' || window.ISLOG) {
      console.warn(...log)
    }
  },
  error(log){
    if (process.env.NODE_ENV === 'development' || window.ISLOG) {
      console.error(...log)
    }
  }
};
export default LOG


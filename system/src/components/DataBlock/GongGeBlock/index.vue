<!--
*@gong-ge-block
*@author yuge
*@date 2023/7/4 20:54
-->
<template>
  <div class="gong-ge-block-main">
    <div class="gong-ge-block" v-for="(item,i) in valueList"
         @click="selectChange(item)" :class="type">
      <img :key="i" :src="selected.title===item.title?iconSelectedList[i]:iconList[i]" alt="">
      <div class="title">{{ item.title }}</div>
    </div>

    <div class="img-center" :class="type">
      <img :src="iconCenter" alt="">
    </div>
  </div>
</template>

<script>
export default {
  name: "gong-ge-block",
  props: {
    //类型 founder or radio
    type: {type: String, default: 'founder'},

    //必须有title
    valueList: {
      type: Array,
      default: () => [
        {value: '', title: '安置帮教'},
        {value: '', title: '吸毒'},
        {value: '', title: '精神病患者'},
        {value: '', title: '社区矫正'},
      ]
    },

    //v-model
    value: {required: true}
  },
  components: {},
  data() {
    return {
      selected: {title: ''}
    }
  },
  watch: {
    value: {
      handler(v) {
        this.selected = v || {}
      },
      immediate: true
    },
    selected(v) {
      this.$emit('input', v)
    },
  },
  computed: {
    iconList() {
      const pre = './img/' + this.type
      return [
        require(pre + '/1.png'),
        require(pre + '/2.png'),
        require(pre + '/3.png'),
        require(pre + '/4.png'),
      ]
    },
    iconSelectedList() {
      const pre = './img/' + this.type
      return [
        require(pre + '/1-selected.png'),
        require(pre + '/2-selected.png'),
        require(pre + '/3-selected.png'),
        require(pre + '/4-selected.png'),
      ]
    },
    iconCenter() {
      const pre = './img/' + this.type
      return require(pre + '/center.png')
    }
  },
  mounted() {
  },
  methods: {
    selectChange(item) {
      this.selected = item
      this.$emit('selectChange', item)
    }
  },
  destroyed() {
  }
}
</script>

<style scoped lang="scss">
.gong-ge-block-main {
  height: 100%;
  width: 100%;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  padding: 0 8px;
  position: relative;

  .gong-ge-block {
    width: calc(50% - 8px);
    height: calc(50% - 20px);
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 51;
    cursor: pointer;
    margin-top: 16px;

    img {
      position: absolute;
      top: 0;
      left: 0;
      height: 100%;
      width: 100%;
      object-fit: fill;
    }

    .title {
      color: white;
      font-size: 18px;
      font-weight: bold;
    }
  }



  .img-center {
    position: absolute;
    top: 3%;
    left: 0;
    height: 100%;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 50;
    pointer-events: none;

    img {
      width: 14%;
    }
  }

  .img-center.radio {
    img {
      margin-top: 3px;
      margin-right: 2px;
      width: 13%;
    }
  }
}
</style>

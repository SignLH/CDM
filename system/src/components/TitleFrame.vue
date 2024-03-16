<template>
  <div class="title-frame-view-2">
    <div class="frame-title-view" :class="mode">
      <div class="title">{{ title }}</div>
      <div class="decorate"></div>
      <div class="action">
        <slot name="action"></slot>
      </div>
    </div>
    <div class="frame-content-view">
      <div class="frame-content-inner">
        <slot></slot>
      </div>
    </div>
    <div class="frame-cube"></div>
    <div class="frame-ltt-line"></div>
    <div class="frame-lbb-line"></div>
    <div class="frame-ltl-line">
      <div class="l1"></div>
      <div class="l2"></div>
      <div class="l3"></div>
      <div class="l4"></div>
    </div>
    <div class="frame-rtt-line"></div>
    <div class="frame-rtr-line"></div>
  </div>
</template>

<script>

export default {
  props: {
    title: {
      type: String,
      default: '标题'
    },
    mode: {
        type: String,
        default: ''
    }
  },
  data() {
    return {}
  },
  computed: {},
  created() {

  },
  methods: {}
}
</script>

<style lang="scss" scoped>
.title-frame-view-2 {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border-right: 2px solid #05539E;
  border-bottom: 2px solid #05549e8e;
  overflow: visible;

  .frame-title-view {
    height: 35px;
    width: 100%;
    background: linear-gradient(to right, #0875ceb9 20%, #ffffff00 80%);
    border-top: 2px solid #186fe236;

    display: flex;
    align-items: center;
    justify-content: flex-start;

    position: relative;

    .title {
      padding-left: 15px;
      font-size: 24px;
      color: #ffffff;
      font-family: YouSheBiaoTiHei;
      line-height: 35px;
      z-index: 50;
    }

    .decorate {
      //position: absolute;
      //top: 10px;
      //left: 150px;
      margin-left: 16px;
      width: 120px;
      height: 16px;
      background: linear-gradient(90deg, #1898e296 50%, transparent 50%);
      background-size: 6px 100%;
      transform: skewX(-35deg);
    }

    .action {
      position: absolute;
      top: 6px;
      right: 6px;
      padding-right: 10px;
    }
  }
  .frame-title-view.red{
    background: linear-gradient(to right, #ce1508b9 20%, #ffffff00 80%);
  }

  .frame-content-view {
    width: 100%;
    height: calc(100% - 35px);
    padding: 4px;

    .frame-content-inner {
      width: 100%;
      height: 100%;
      background: #00000013;
      background: linear-gradient(310deg, transparent 10px, #00000013 0);
      box-shadow: 0px 0px 0px 0.5px #146dbb;
      position: relative;
      overflow: hidden;
    }
  }

  .frame-cube {
    position: absolute;
    bottom: -2px;
    right: -2px;
    width: 0;
    height: 0;
    border-bottom: 12px solid #0099ff;
    border-left: 12px solid transparent;
  }

  .frame-ltt-line {
    width: 120px;
    height: 0;
    border-top: 2px solid #0081FF;
    position: absolute;
    top: -1px;
    left: -1px;
  }

  .frame-lbb-line {
    width: 60px;
    height: 0;
    border-top: 2px solid #0081FF;
    position: absolute;
    bottom: -2px;
    left: -1px;
  }

  .frame-ltl-line {
    width: 0;
    display: flex;
    flex-direction: column;
    height: calc(100% + 3px);
    position: absolute;
    top: -1px;
    left: -2px;

    .l1 {
      flex-shrink: 0;
      height: 50px;
      border-left: 2px solid #0080ffd8;
    }

    .l2 {
      flex-shrink: 0;
      height: 50px;
      border-left: 2px dotted #0080ffd8;
    }

    .l3 {
      flex: 1;
      border-left: 2px solid #186fe236;
    }

    .l4 {
      flex-shrink: 0;
      height: 40px;
      border-left: 2px solid #0081FF;
    }
  }

  .frame-rtt-line {
    width: 60px;
    height: 0;
    border-top: 2px solid #0081FF;
    position: absolute;
    top: -1px;
    right: -1px;
  }

  .frame-rtr-line {
    width: 0;
    height: 36px;
    border-right: 2px solid #0081FF;
    position: absolute;
    top: -1px;
    right: -2px;
  }
}
</style>

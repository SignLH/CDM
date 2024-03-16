<!--
*@CityImageBlock
*@author GYY
*@date 2033/8/28 15:19
-->
<template>
  <div class="city-image-block-main" :class="type">
    <img :src="require(`./img/${type}.png`)" v-if="showImg" alt=""/>
    <slot v-if="useSlot"></slot>
    <div class="value" v-else>
      <unit-number :number="value" :unit="unit" :number-size="numSize" :unit-size="unitSize"/>
    </div>
    <div class="title word-ellipsis" :title="title">{{ title }}</div>
  </div>
</template>

<script>
import UnitNumber from "@/components/UnitNumber.vue";

export default {
  name: "CityImageBlock",
  props: {
    title: {type: String, default: '测试标题'},
    value: {type: [String, Number], default: 0},
    //普通类型：disc leaves people tech tech-big tray contracted
    //垃圾分类：ljg ljs ljhs ljyh
    type: {type: String, default: 'disc'},
    unit: {type: String, default: ''},
    useSlot: {type: Boolean},
  },
  data() {
    return {}
  },
  components: {UnitNumber},
  computed: {
    numSize() {
      switch (this.type) {
        case 'people':
          return 25
        case 'tech':
          return 20
        case 'leaves':
          return 28
        case 'tray':
          return 50
        case 'disc':
          return 39
        default:
          return 30
      }
    },
    unitSize() {
      switch (this.type) {
        case 'people':
          return 16
        case 'tech':
          return 14
        case 'tray':
          return 22
        case 'disc':
          return 20
        default:
          return 16
      }
    },
    showImg() {
      switch (this.type) {
        case 'contracted':
          return false
        default:
          return true
      }
    }
  },
  mounted() {
  },
  methods: {},
}
</script>

<style lang='scss' scoped>

.city-image-block-main {
  height: 140px;
  width: 49%;
  margin: 8px 4px;
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  color: #FFFFFF;

  img {
    z-index: -1;
  }

  .value {
    flex: 1;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: flex-start;
  }

  .title {
    width: 100%;
    font-size: 16px;
    text-align: center;
    color: #fff;
  }
}

.contracted {
  height: 100px;
  width: auto;
  padding: 0 20px;
  border-radius: 4px;
  border-top: #4d9cb4 2px solid;
  box-shadow: 0 4px 4px rgba(13, 243, 216, 0.13);
  background-image: linear-gradient(180deg, #093057, transparent 70%);

  .value {
    align-items: center;
  }

  .title {
    padding-bottom: 20px;
  }
}

.ljg, .ljs, .ljhs, .ljyh {
  height: 100px;
  padding-left: 10%;
  box-sizing: border-box;
  .value{
    align-items: flex-end;
  }
  .title{
    padding: 15px 0;
  }
  img {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 40px;
    height: 90%;
    width: 50%;
    object-fit: contain;
  }

}

.leaves {
  padding-top: 20px;
  img {
    height: 55%;
    width: 100%;
    object-fit: contain;
  }
}

.percent {
  .value {
    margin-top: 10px;
    align-items: center;
  }

  img {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 40px;
    height: 90%;
    width: 100%;
    object-fit: contain;
  }
}

.tech {
  height: 70px;

  img {
    position: absolute;
    top: 0;
    left: 0;
    bottom: -22px;
    height: 100%;
    width: 100%;
  }

  .value {
    align-items: flex-end;
  }

  .title {
    height: 50%;
    width: 100%;
    font-size: 14px;
    word-break: break-word;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
}

.tech-big {
  height: 100px;

  img {
    position: absolute;
    top: 0;
    left: 0;
    bottom: -22px;
    height: 100%;
    width: 100%;
  }

  .value {
    align-items: flex-end;
  }

  .title {
    height: 40px;
    font-size: 22px;
  }
}

.people {
  padding-top: 15px;
  img {
    position: absolute;
    left: 0;
    top: 40px;
    bottom: 40px;
    height: 55%;
    width: 100%;
    object-fit: contain;
  }
}

.tray {
  img {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 40px;
    height: 80%;
    width: 100%;
    object-fit: contain;

  }
}

.disc {
  .value {
    margin-top: 5px;
  }

  img {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 40px;
    height: 80%;
    width: 100%;
    object-fit: contain;

  }
}
</style>

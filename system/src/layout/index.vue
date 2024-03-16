<template>
  <div class="layout-view">
    <div class="map-view" ref="base-map-container">
      <relationship-chart/>
    </div>

    <div class="loading-view"></div>
    <div class="top-view">
      <img src="@/assets/layout/top-back.png" class="top-back-img" alt=""/>
      <div class="top-weather"></div>
      <div class="top-title">APSP认知诊断系统</div>
      <div class="top-time-setting">
        <div class="time">
          <div class="top">{{ currentTime | formatDate('hh:mm:ss') }}</div>
          <div class="btm">{{ currentTime | formatDate('yyyy-MM-dd 星期W') }}</div>
        </div>
        <div class="setting">
          <img src="@/assets/layout/setting.png" class="icon" alt=""/>
        </div>
      </div>
    </div>
    <div class="content-view">
      <router-view/>
    </div>


    <div class="left-word">
      问题ID：
      <el-select v-model="questionId" placeholder="请选择" size="mini" style="width: 70px">
        <el-option
            v-for="item in questionList"
            :key="item.id"
            :label="item.id"
            :value="item.id">
        </el-option>
      </el-select>
    </div>
    <div class="right-word">
      <div class="flex-around">
        学生ID：
        <el-select v-model="studentId" placeholder="请选择" size="mini" style="width: 120px">
          <el-option
              v-for="item in studentList"
              :key="item.id"
              :label="item.id"
              :value="item.id">
          </el-option>
        </el-select>
      </div>
      <custom-button style="margin-top: 8px" @click.native="isDialogShow=!isDialogShow"
      >进入知识空间</custom-button>
    </div>

    <student-dialog v-model="isDialogShow"/>
  </div>
</template>
<script>
import {mapGetters} from 'vuex'
import RelationshipChart from "@/components/RelationshipChart/index.vue";
import CustomButton from "@/components/custom-button.vue";
import StudentDialog from "@/layout/components/StudentDialog.vue";

export default {
  components: {
    StudentDialog,
    CustomButton,
    RelationshipChart
  },
  data() {
    return {
      isDialogShow:false,
      questionId:1,
      questionList:[],
      studentId:'',
      studentList:[],

    }
  },
  computed: {
    //联勤联动： lqld ; 智慧环卫： zhhw ;
    //智慧社区： zhsq ; 文明创建： wmcj ;
    ...mapGetters([
      'currentTime'
    ])
  },
  mounted() {
    this.init()
    this.$store.dispatch('time/startTimeInterval')
  },
  methods: {
    init() {
      const res=[]
      for (let i = 1; i <= 50; i++) {
        res.push({id:i})
      }
      const stu=[]
      for (let i = 1; i <= 246; i++) {
        stu.push({id:i+20232000})
      }
      this.questionList=res
      this.studentList=stu
      this.studentId=20232002
    }
  },
  destroyed() {
    this.$store.dispatch('time/clearTimeInterval')
  },
}
</script>
<style lang="scss" scoped>
.layout-view {
  width: 100%;
  height: 100%;
  position: relative;


  .map-view {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 0;

    img {
      width: 100%;
      height: 100%;
    }
  }

  .loading-view {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: -1;

    color: white;
    background-color: #000E28;
  }

  .loading-view::after {
    position: fixed;
    top: 0;
    left: 0;
    content: '';
    width: 100%;
    height: 100%;
    background-image: linear-gradient(0deg,
        #9FF7EA 0px,
        #9FF7EA 1px,
        transparent 1px,
        transparent 100px),
    linear-gradient(90deg,
            #9FF7EA 0px,
            #9FF7EA 1px,
            transparent 1px,
            transparent 100px);
    background-size: 50px 50px;
    opacity: 0.1;
    z-index: -1;
  }


  .top-view {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100px;
    background: linear-gradient(to bottom, #012d52e3 45%, #01233f00);
    display: flex;
    align-items: center;
    z-index: 2000;
    overflow: visible;

    .top-back-img {
      z-index: -1;
      position: absolute;
      top: -5px;
      left: 0;
      width: 100%;
      height: calc(100% + 5px);
    }

    .top-title {
      flex: 1;
      text-align: center;
      font-size: 40px;
      font-family: YouSheBiaoTiHei;
      color: #ffffff;
      padding-bottom: 30px;
    }

    .top-weather {
      width: 300px;
      flex-shrink: 0;
      display: flex;
      align-items: flex-end;
      color: #ffffff;
      padding-left: 25px;
      padding-bottom: 8px;
      cursor: pointer;

      .icon {
        width: 30px;
        height: 30px;
        margin-right: 10px;
      }

      .number {
        font-size: 28px;
        line-height: 1;
      }

      .unit {
        font-size: 15px;
      }

      .tip {
        font-size: 15px;
        margin-left: 10px;
      }
    }

    .top-time-setting {
      width: 300px;
      flex-shrink: 0;
      padding-bottom: 5px;
      display: flex;
      align-items: stretch;
      justify-content: flex-end;

      .time {
        display: flex;
        flex-direction: column;
        padding-right: 20px;
        color: #ffffff;

        .top {
          font-size: 22px;
          line-height: 1;
          font-family: YouSheBiaoTiHei;
        }

        .btm {
          font-size: 11px;
          line-height: 1;
        }
      }

      .setting {
        width: 70px;
        padding: 0 20px;
        border-left: 2px solid #008afc9d;
        display: flex;
        align-items: center;

        .icon {
          width: 23px;
          height: 23px;
          cursor: pointer;
        }

        .icon:hover {
          width: 24px;
          height: 24px;
          cursor: pointer;
        }
      }

      .szry {
        width: 34px;
        height: 34px;
        cursor: pointer;
        margin-right: 25px;
      }
    }

  }

  .left-word {
    z-index: 1;
    color: white;
    font-size: 20px;
    position: absolute;
    top: 10%;
    left: 410px;
    display: flex;
    align-items: center;
  }

  .right-word {
    z-index: 1;
    color: white;
    font-size: 20px;
    position: absolute;
    top: 10.5%;
    right: 410px;
  }

  .content-view {
    position: absolute;
    top: 100px;
    left: 0;
    width: 100%;
    height: calc(100% - 100px);
    z-index: 1000;
    pointer-events: none;

    .frame-view {
      position: absolute;
      width: 520px;
      height: 100%;
      top: 0;
      z-index: 1100;
      background: #021C31DD;
      box-shadow: 0 0 60px 60px #021C31DD;
      pointer-events: auto;
    }

    .frame-view.left {
      left: -520px;
    }

    .frame-view.right {
      right: -520px;
    }


    .page-view {
      position: relative;
      width: 100%;
      top: -5px;
      height: calc(100% - 15px);
      z-index: 1500;
      overflow: hidden;
    }

    .map-nav {
      position: absolute;
      right: 10px;
      border: 1px solid #146dbb;
      z-index: 2000;
      pointer-events: auto;
      overflow: hidden;

      .map-nav-item {
        width: 46px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #02233fb6;

        .icon {
          width: 18px;
          height: auto;
        }
      }

      .map-nav-item + .map-nav-item {
        border-top: 1px solid #146dbb;
      }

      .map-nav-item:hover {
        cursor: pointer;

        .icon {
          width: 20px;
          height: auto;
        }
      }
    }

    .map-nav.map-nav1 {
      top: 0;
    }

    .map-nav.map-nav2 {
      bottom: 20px; //79
    }

    .area-nav {
      position: absolute;
      top: 0;
      left: 10px;
      z-index: 2000;
      border: 1px solid #146dbb;
      pointer-events: auto;


      .area-nav-item {
        width: 46px;
        height: 46px;
        font-family: YouSheBiaoTiHei;
        color: #ffffff;
        font-size: 22px;
        position: relative;
        text-align: center;
        line-height: 46px;
        background: #02233fb6;
        overflow: visible;

        .item-big {
          z-index: 2001;
          text-align: left;
          background: linear-gradient(to right, #02233f, #02233f56);
          position: absolute;
          top: 0;
          left: 0;
          height: 100%;
          letter-spacing: 3px;
          width: 160px;
          padding-left: 15px;
          display: none;
          white-space: nowrap;
          border-radius: 0 10px 10px 0;
        }
      }

      .area-nav-item + .area-nav-item {
        border-top: 1px solid #146dbb;
      }

      .area-nav-item:hover {
        cursor: pointer;

        .item-big {
          display: inline-block;
        }
      }
    }

    .direction-nav {
      position: absolute;
      bottom: 20px; //79
      left: 10px;
      width: 100px;
      height: 100px;
      border-radius: 50%;
      box-shadow: 0 0 10px #022438;
      z-index: 2000;
      pointer-events: auto;
      background: radial-gradient(#ffffff00 10px, #0c71f5e1);

      .center {
        width: 40px;
        height: 40px;
        border: 1px solid #0080ff94;
        position: absolute;
        border-radius: 50%;
        left: calc(50% - 20px);
        top: calc(50% - 20px);
        display: flex;
        align-items: center;
        justify-content: center;

        img {
          width: 22px;
          height: 22px;
        }
      }

      .to {
        width: 28px;
        height: 28px;
        position: absolute;
        cursor: pointer;
      }

      .to.N {
        top: 0;
        left: calc(50% - 14px);
      }

      .to.N:hover {
        top: -2px;
      }

      .to.S {
        bottom: 0;
        left: calc(50% - 14px);
      }

      .to.S:hover {
        bottom: -2px;
      }

      .to.W {
        left: 0;
        top: calc(50% - 14px);
      }

      .to.W:hover {
        left: -2px;
      }

      .to.E {
        right: 0;
        top: calc(50% - 14px);
      }

      .to.E:hover {
        right: -2px;
      }

    }

    .page-nav {
      width: 520px;
      position: absolute;
      bottom: 20px;
      right: calc(50% - 260px);
      z-index: 2000;
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: auto;
      flex-wrap: wrap;


      .page-nav-item {
        width: 120px;
        height: 42px;
        transform: skewX(-30deg);
        border-radius: 3px;
        margin-bottom: 4px;
        margin-top: 8px;

        border: 3px solid transparent;
        background-clip: padding-box, border-box;
        background-origin: padding-box, border-box;
        background-image: linear-gradient(to right, #05488bc5, #05488b96), linear-gradient(to right, #56acf1, #578aef17);

        .inner {
          width: 100%;
          height: 100%;
          display: flex;
          align-items: flex-end;
          justify-content: center;
          transform: skewX(30deg);
          padding-bottom: 8px;

          .icon {
            height: 20px;
            width: auto;
            margin-right: 3px;
          }

          .label {
            font-size: 17px;
            letter-spacing: 1px;
            color: #ffffff;
            font-style: normal;
          }
        }
      }

      .page-nav-item + .page-nav-item {
        margin-left: 10px;
      }

      .page-nav-item:hover,
      .page-nav-item.choose {
        cursor: pointer;
        background-image: linear-gradient(to right, #0080FF, #0080ffc9), linear-gradient(to right, #00a2ff, #578aef17);
      }

      .page-nav-item:first-child {
        border-radius: 3px 3px 3px 12px;
      }

      .page-nav-item:last-child {
        border-radius: 3px 12px 3px 3px;
      }
    }


  }

  .open-view {
    position: fixed;
    z-index: 3000;
    top: calc(50% - 70px);
    height: 140px;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;

    .back-img {
      position: absolute;
      top: 0;
      width: 100%;
      height: 100%;
    }

    .icon {
      width: 18px;
      height: auto;
    }
  }

  .open-view.left {
    left: 0;

    .back-img {
      left: 0;
    }
  }

  .open-view.right {
    right: 0;

    .back-img {
      right: 0;
    }
  }

  .open-view:hover {
    cursor: pointer;

    .icon {
      width: 20px;
    }
  }

}
</style>
<style>
.page-left-view {
  pointer-events: auto;
}

.page-right-view {
  pointer-events: auto;
}
</style>
<style lang="scss">
.custom-poi-main {
  position: absolute;
  top: 0;
  left: 0;
  height: 30px;
  width: 100px;
  z-index: 99999;

  .inner {
    height: 100%;
    width: 100%;
    background-color: #0a5fa4;
    border-radius: 32px;
    border: 1px #0a5fa4 solid;
    color: white;
  }
}
</style>

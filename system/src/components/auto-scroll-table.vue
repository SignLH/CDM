<!--
*@auto-scroll-table
*@author GYY
*@date 2023/3/20 9:22
-->
<template>
  <div class="auto-scroll-table-main">
    <div class="table-head-view table-head">
      <div class="table-head-item" v-for="(item,i) in tableHeads" :key="i"
           :title="item.name">{{ item.name }}</div>
    </div>

    <div class="table-list">
      <auto-scroll-list :list-data="tableList" v-if="tableList&&tableList.length > 0">
        <div class="table-line-out-view">
          <div class="table-line-view" v-for="(item,i) in tableList" :key="i">
            <div class="table-line-item" v-for="(key,j) in tableHeads" :key="j" :title="item[key.props]">
              <template v-if="key.type==='slot'">
                <slot :name="key.props" :data="item[key.props]"></slot>
              </template>
<!--              tableIndex类型显示序号-->
              <template v-else-if="key.props==='tableIndex'">{{i+1}}</template>
              <template v-else>{{ item[key.props] }}</template>
            </div>
          </div>
        </div>
      </auto-scroll-list>
      <div class="table-no-data" v-else>暂无数据</div>
    </div>

  </div>
</template>

<script>
import AutoScrollList from "@/components/AutoScrollList";

export default {
  name: "auto-scroll-table",
  props: {
    tableList: {
      type: Array, default: () => [
        {item1: '内容1', item2: '内容2', item3: '内容3', item4: '内容4'},
        {item1: '内容1', item2: '内容2', item3: '内容3', item4: '内容4'},
        {item1: '内容1', item2: '内容2', item3: '内容3', item4: '内容4'},
        {item1: '内容1', item2: '内容2', item3: '内容3', item4: '内容4'},
        {item1: '内容1', item2: '内容2', item3: '内容3', item4: '内容4'},
      ]
    },
    tableHeads: {
      //内置props tableIndex
      type: Array, default: () => [
        {props: 'item1', name: '标题1'},
        {props: 'item2', name: '标题2'},
        {props: 'item3', name: '标题3'},
        {props: 'item4', name: '标题4'},
      ]
    },
  },
  data() {
    return {}
  },
  watch: {},
  components: {AutoScrollList},
  mounted() {
  },
  methods: {},
}
</script>

<style lang='scss' scoped>
.auto-scroll-table-main {
  height: 100%;
  width: 100%;

  overflow: hidden;
  font-size: 14px !important;

  display: flex;
  flex-direction: column;

  .table-head {
    flex-shrink: 0;
    width: 100%;
  }

  .table-list {
    width: 100%;
    flex: 1;
    position: relative;
    overflow: hidden;
  }
}
</style>

<template>
  <div id="app">
    <router-view/>
  </div>
</template>
<script>
import {InitTimingTask} from "@/utils/global-work";

export default {
  computed: {},
  created(){
    InitTimingTask()
  },
  mounted() {
  },
  methods: {}
}
</script>
<style>

</style>

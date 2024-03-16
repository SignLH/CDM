<template>
    <div class="spot-label-view">
        <div class="spot" v-if="spotShow"></div>
        <div class="label" :style="`font-size:${size}px`">{{ label }}</div>
    </div>
</template>

<script>
export default {
    props: {
        label: {
            type: String,
            default: '标题'
        },
        size: {
            type: String, Number,
            default: '15'
        },
        spotShow: {
            type: Boolean,
            default: true
        }
    },
}
</script>

<style lang="scss" scoped>
.spot-label-view {

    display: flex;
    align-items: center;

    .spot {
        width: 6px;
        height: 6px;
        background: #46FEFF;
        margin-right: 5px;
        line-height: 1;
    }

    .label {
        color: #46FEFF;
        font-size: 15px;
    }

}
</style>
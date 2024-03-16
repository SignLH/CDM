<template>
    <div class="unit-number-view">
        <div class="number" :style="`font-size:${numberSize}px`">{{ number }}</div>
        <div class="unit" :style="`font-size:${unitSize}px`">{{ unit }}</div>
    </div>
</template>

<script>

export default {
    props: {
        number: {
            type: [String, Number],
            default: ''
        },
        numberSize: {
            type: [String, Number],
            default: '18'
        },
        unit: {
            type: String,
            default: ''
        },
        unitSize: {
            type: [String, Number],
            default: '11'
        },
    },
}
</script>

<style lang="scss" scoped>
.unit-number-view {

    display: flex;
    align-items: flex-end;

    .number {
        line-height: 1;
        font-family: Rajdhani-SemiBold;
    }

}
</style>
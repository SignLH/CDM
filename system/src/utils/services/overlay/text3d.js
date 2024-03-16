import {SuffixForOverlayId} from "@/utils/services/overlay/entity";
import { Material, Text3d, Text3dBevelType } from '@sheencity/diva-sdk';
import { Vector3 } from '@sheencity/diva-sdk-math';
import LOG from "@/utils/self-log";

export async function addText3d(client,id,textContent='test',coordinate) {
  const text3d = new Text3d({
    name: id,
    group: SuffixForOverlayId.getGroupValueById(id),
    visible: true,
    clickable: true,
    coordinate,
    rotation: new Vector3(0,-50,0),
    scale: new Vector3(150, 150, 150),
    text: {
      content: textContent,
      align: 'center',
      verticalAlign: 'middle',
      letterSpacing: 10,
      wordSpacing: 0,
      lineSpacing: 0,
    },
    style: '自发光',
    material: new Material({
      color: '#00c409',
      emissive: 1.0,
      opacity: 0,
      speed: 0.2,
    }),
    thickness: 10,
    outline: false,
    bevel: {
      size: 0,
      type: Text3dBevelType.Linear,
      material: new Material({
        color: '#00b0fd',
        emissive: 1.0,
        opacity: 0,
        speed: 0.2,
      }),
    },
  });

  try {
    await text3d.setClient(client);
  }catch (e) {
    LOG.warn(['create text3d failed!',e])
  }
}


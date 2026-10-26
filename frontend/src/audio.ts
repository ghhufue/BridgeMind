import { Howl, Howler } from "howler";

const sounds = {
  card: new Howl({ src: ["/sounds/card.wav"], volume: 0.35, preload: true }),
  call: new Howl({ src: ["/sounds/call.wav"], volume: 0.22, preload: true }),
  trick: new Howl({ src: ["/sounds/trick.wav"], volume: 0.22, preload: true }),
};
export const setSound = (enabled: boolean) => {
  Howler.mute(!enabled);
};
export const playSound = (kind: keyof typeof sounds) => {
  sounds[kind].play();
};

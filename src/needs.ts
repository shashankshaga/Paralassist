export type NeedId =
  | "help"
  | "water"
  | "pain"
  | "bathroom"
  | "yes"
  | "no"
  | "food"
  | "rest";

export type Need = {
  id: NeedId;
  label: string;
  phrase: string;
  hz: number;
  accent: string;
};

export const NEEDS: Need[] = [
  { id: "help", label: "Help", phrase: "I need help now.", hz: 15, accent: "#ff5d4f" },
  { id: "water", label: "Water", phrase: "I need water.", hz: 12, accent: "#4fd0ff" },
  { id: "pain", label: "Pain", phrase: "I am in pain.", hz: 10, accent: "#ffb020" },
  { id: "bathroom", label: "Bathroom", phrase: "I need the bathroom.", hz: 8.57, accent: "#7ad0a4" },
  { id: "yes", label: "Yes", phrase: "Yes.", hz: 13.5, accent: "#3ee0b2" },
  { id: "no", label: "No", phrase: "No.", hz: 7.5, accent: "#c9b8ff" },
  { id: "food", label: "Food", phrase: "I am hungry.", hz: 11, accent: "#f0c36a" },
  { id: "rest", label: "Rest", phrase: "I want to rest.", hz: 9, accent: "#9eb6c8" },
];

export const FS = 256;
export const WINDOW_SEC = 1;
export const WINDOW_SAMPLES = FS * WINDOW_SEC;

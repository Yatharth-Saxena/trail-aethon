/**
 * Maps detected objects to generic aerospace-grade representative SVG images.
 */
export function getObjectVisual(obj) {
  if (!obj) return null;

  const rawStr = (
    typeof obj === "string"
      ? obj
      : (obj.raw_label || obj.class_label || obj.label || obj.display_name || "")
  ).toLowerCase().trim();

  if (
    !rawStr ||
    rawStr === "none" ||
    rawStr === "unknown" ||
    rawStr.includes("monitoring") ||
    rawStr === "seated" ||
    rawStr === "standing" ||
    rawStr === "active"
  ) {
    return null;
  }

  // Match recognized classes
  if (rawStr.includes("phone") || rawStr.includes("mobile") || rawStr.includes("cell")) {
    return { src: "/objects/phone.svg", name: "Cell Phone", badge: "DEVICE" };
  }
  if (rawStr.includes("bottle") || rawStr.includes("flask")) {
    return { src: "/objects/bottle.svg", name: "Bottle", badge: "CONTAINER" };
  }
  if (rawStr.includes("cup") || rawStr.includes("mug")) {
    return { src: "/objects/cup.svg", name: "Cup", badge: "CONTAINER" };
  }
  if (rawStr.includes("laptop") || rawStr.includes("computer")) {
    return { src: "/objects/laptop.svg", name: "Laptop", badge: "TERMINAL" };
  }
  if (rawStr.includes("mouse")) {
    return { src: "/objects/mouse.svg", name: "Mouse", badge: "PERIPHERAL" };
  }
  if (rawStr.includes("keyboard")) {
    return { src: "/objects/keyboard.svg", name: "Keyboard", badge: "INPUT" };
  }
  if (rawStr.includes("scissors") || rawStr.includes("shears")) {
    return { src: "/objects/scissors.svg", name: "Scissors / Shears", badge: "TOOL" };
  }
  if (rawStr.includes("book") || rawStr.includes("manual") || rawStr.includes("binder") || rawStr.includes("notebook")) {
    return { src: "/objects/book.svg", name: "Flight Manual", badge: "DOCUMENT" };
  }
  if (rawStr.includes("clock") || rawStr.includes("timer") || rawStr.includes("watch")) {
    return { src: "/objects/clock.svg", name: "Chronometer", badge: "TIMER" };
  }
  if (rawStr.includes("remote")) {
    return { src: "/objects/remote.svg", name: "Remote Control", badge: "CONTROL" };
  }
  if (rawStr.includes("pen") || rawStr.includes("pencil") || rawStr.includes("marker") || rawStr.includes("stylus")) {
    return { src: "/objects/pen.svg", name: "Stylus / Pen", badge: "TOOL" };
  }
  if (rawStr.includes("bowl") || rawStr.includes("dish")) {
    return { src: "/objects/bowl.svg", name: "Lab Bowl", badge: "CONTAINER" };
  }
  if (rawStr.includes("knife") || rawStr.includes("scalpel") || rawStr.includes("cutter")) {
    return { src: "/objects/knife.svg", name: "Precision Cutter", badge: "TOOL" };
  }
  if (rawStr.includes("fork")) {
    return { src: "/objects/fork.svg", name: "Fork", badge: "UTENSIL" };
  }
  if (rawStr.includes("spoon")) {
    return { src: "/objects/spoon.svg", name: "Spoon", badge: "UTENSIL" };
  }
  if (rawStr.includes("chair") || rawStr.includes("couch") || rawStr.includes("seat")) {
    return { src: "/objects/chair.svg", name: "Station Chair", badge: "FURNITURE" };
  }
  if (rawStr.includes("tv") || rawStr.includes("monitor") || rawStr.includes("screen")) {
    return { src: "/objects/monitor.svg", name: "HUD Display", badge: "DISPLAY" };
  }
  if (rawStr.includes("backpack") || rawStr.includes("pack")) {
    return { src: "/objects/backpack.svg", name: "EVA Backpack", badge: "GEAR" };
  }
  if (rawStr.includes("bag") || rawStr.includes("handbag") || rawStr.includes("suitcase")) {
    return { src: "/objects/bag.svg", name: "Payload Kit", badge: "CARRIER" };
  }
  if (rawStr.includes("umbrella")) {
    return { src: "/objects/umbrella.svg", name: "Umbrella", badge: "ACCESSORY" };
  }
  if (rawStr.includes("plant") || rawStr.includes("potted plant")) {
    return { src: "/objects/plant.svg", name: "Botany Sample", badge: "BIOLOGICAL" };
  }
  if (rawStr.includes("apple")) {
    return { src: "/objects/apple.svg", name: "Apple", badge: "RATION" };
  }
  if (rawStr.includes("banana")) {
    return { src: "/objects/banana.svg", name: "Banana", badge: "RATION" };
  }
  if (rawStr.includes("orange")) {
    return { src: "/objects/orange.svg", name: "Orange", badge: "RATION" };
  }
  if (rawStr.includes("pizza")) {
    return { src: "/objects/pizza.svg", name: "Pizza", badge: "RATION" };
  }
  if (rawStr.includes("sandwich")) {
    return { src: "/objects/sandwich.svg", name: "Sandwich", badge: "RATION" };
  }
  if (rawStr.includes("donut")) {
    return { src: "/objects/donut.svg", name: "Donut", badge: "RATION" };
  }
  if (rawStr.includes("cake")) {
    return { src: "/objects/cake.svg", name: "Cake", badge: "RATION" };
  }
  if (rawStr.includes("ball") || rawStr.includes("sports ball")) {
    return { src: "/objects/ball.svg", name: "Ball", badge: "CALIBRATION" };
  }
  if (rawStr.includes("teddy") || rawStr.includes("bear")) {
    return { src: "/objects/teddy.svg", name: "Zero-G Indicator", badge: "INDICATOR" };
  }
  if (rawStr.includes("toothbrush")) {
    return { src: "/objects/toothbrush.svg", name: "Toothbrush", badge: "HYGIENE" };
  }
  if (rawStr.includes("hair drier") || rawStr.includes("hairdryer")) {
    return { src: "/objects/hairdrier.svg", name: "Thermal Blower", badge: "TOOL" };
  }
  if (rawStr.includes("vase")) {
    return { src: "/objects/vase.svg", name: "Specimen Vase", badge: "CONTAINER" };
  }
  if (rawStr.includes("tie")) {
    return { src: "/objects/tie.svg", name: "Uniform Tie", badge: "APPAREL" };
  }
  if (rawStr.includes("object a") || rawStr.includes("red block") || rawStr.includes("block a")) {
    return { src: "/objects/block_red.svg", name: "Object A (Red Block)", badge: "PAYLOAD" };
  }
  if (rawStr.includes("object b") || rawStr.includes("wooden block") || rawStr.includes("block b")) {
    return { src: "/objects/block_wood.svg", name: "Object B (Base Block)", badge: "PAYLOAD" };
  }
  if (rawStr.includes("tray")) {
    return { src: "/objects/tray.svg", name: "Assembly Tray", badge: "APPARATUS" };
  }
  if (rawStr.includes("button")) {
    return { src: "/objects/button.svg", name: "Complete Button", badge: "CONTROL" };
  }

  // Generic fallback for any other recognized object
  const displayName = typeof obj === "object" ? (obj.display_name || obj.label || "Component") : obj;
  return { src: "/objects/component.svg", name: displayName, badge: "HARDWARE" };
}

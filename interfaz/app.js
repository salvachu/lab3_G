const el = id => document.getElementById(id);
const primary = el("primary"), secondary = el("secondary"), seed = el("seed");
let names = {}, history = [];
function secondType() {
  const old = secondary.value;
  secondary.replaceChildren(new Option("Ninguno", ""));
  for (const [key, name] of Object.entries(names)) {
    if (key !== primary.value) secondary.add(new Option(name, key));
  }
  secondary.value = old === primary.value ? "" : old;
}
function show(result) {
  el("specimen").textContent = result.types.map(t => names[t]).join(" / ");
  el("number").textContent = result.model.toUpperCase() + " · SEMILLA " + result.seed;
  for (const side of ["front", "back"]) {
    el(side).src = result[side];
    el(side).hidden = false;
    const link = el("save-" + side);
    link.href = result[side];
    link.download = result.model + "-" + result.types.join("-") + "-" + result.seed + "-" + side + ".png";
    link.hidden = false;
  }
}
function selectResult(item) {
  el("model").value = item.model;
  primary.value = item.types[0];
  secondType();
  secondary.value = item.types[1] || "";
  seed.value = item.seed;
  el("fixed").checked = true;
  show(item);
}
async function generate() {
  const submit = el("submit");
  submit.disabled = true;
  el("status").textContent = "Generando…";
  try {
    const response = await fetch("/api/generar", {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({types:[primary.value,secondary.value].filter(Boolean), seed:el("fixed").checked ? Number(seed.value) : null, model:el("model").value})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    seed.value = result.seed;
    show(result);
    history = [result,...history.filter(p => p.model !== result.model || p.seed !== result.seed || p.types.join() !== result.types.join())].slice(0,8);
    el("items").replaceChildren(...history.map(item => {
      const button = document.createElement("button"), img = new Image();
      img.src = item.front; img.alt = "";
      button.append(img, document.createTextNode(item.model+" · "+item.types.map(t=>names[t]).join(" / ")+" · "+item.seed));
      button.onclick = () => selectResult(item);
      return button;
    }));
    document.querySelector(".history").hidden = false;
    el("status").textContent = "";
  } catch (e) { el("status").textContent = "No se pudo generar. " + e.message; }
  finally { submit.disabled = false; }
}
el("form").onsubmit = event => { event.preventDefault(); generate(); };
seed.oninput = () => { el("fixed").checked = true; };
el("random").onclick = () => {
  seed.value = crypto.getRandomValues(new Uint32Array(1))[0] >>> 1;
  el("fixed").checked = true;
};
primary.onchange = secondType;
async function init() {
  names = await (await fetch("/api/tipos")).json();
  for (const [v,n] of Object.entries(names)) primary.add(new Option(n,v));
  primary.value = "water"; secondType(); secondary.value = "dragon";
  await generate();
}
init().catch(() => { el("status").textContent = "No se pudieron cargar los tipos. Recarga la página."; });

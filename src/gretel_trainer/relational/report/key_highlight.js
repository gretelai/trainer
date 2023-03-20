function mouseOverPk(event) {
    event.preventDefault();
    let { currentTarget } = event;
    let target = currentTarget;

    target.classList.add("label__highlighted");
    target.parentNode.parentNode.classList.add("row__highlighted");

    document.querySelectorAll(`[data-fk-from=${target.id}]`).forEach(
        fk => {
            fk.classList.add("label__highlighted");
            fk.parentNode.parentNode.classList.add("row__highlighted");
        }
    )
}

function mouseOutPk(event) {
    event.preventDefault();
    let { currentTarget } = event;
    let target = currentTarget;

    target.classList.remove("label__highlighted");
    target.parentNode.parentNode.classList.remove("row__highlighted");

    document.querySelectorAll(`[data-fk-from=${target.id}]`).forEach(
        fk => {
            fk.classList.remove("label__highlighted");
            fk.parentNode.parentNode.classList.remove("row__highlighted");
        }
    )
}

function mouseOverFk(event) {
    event.preventDefault();
    let { currentTarget } = event;
    let target = currentTarget;

    target.classList.add("label__highlighted");
    target.parentNode.parentNode.classList.add("row__highlighted");

    let pk = document.getElementById(target.getAttribute('data-fk-from'));
    pk.classList.add("label__highlighted");
    pk.parentNode.parentNode.classList.add("row__highlighted");
}

function mouseOutFk(event) {
    event.preventDefault();
    let { currentTarget } = event;
    let target = currentTarget;

    target.classList.remove("label__highlighted");
    target.parentNode.parentNode.classList.remove("row__highlighted");

    let pk = document.getElementById(target.getAttribute('data-fk-from'));
    pk.classList.remove("label__highlighted");
    pk.parentNode.parentNode.classList.remove("row__highlighted");
}

const pks = document.querySelectorAll('[data-pk]');
pks.forEach(pk => { pk.onmouseover = mouseOverPk });
pks.forEach(pk => { pk.onmouseout = mouseOutPk });

const fks = document.querySelectorAll('[data-fk-from]');
fks.forEach(fk => { fk.onmouseover = mouseOverFk });
fks.forEach(fk => { fk.onmouseout = mouseOutFk });

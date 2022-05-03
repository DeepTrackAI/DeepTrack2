import * as React from "react";
import Logo from "../../images/logo.png";

// markup
const Header = () => {
  return (
    <header className={"header root"}>
      <img alt="DeepTrack logo" src={Logo}></img>
    </header>
  );
};

export default Header;

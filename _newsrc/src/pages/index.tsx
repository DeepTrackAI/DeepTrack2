import * as React from "react";
import "./styles.scss";
import { Header } from "./header";
import { TitleCard } from "./titlecard";
import { GetStartedCard } from "./GetStartedCard";
import { createTheme, ThemeProvider } from "@mui/material";

// styles
const theme = createTheme({
  palette: {
    primary: {
      main: "#e76f51",
    },
    secondary: {
      main: "#264653",
    },
  },
  typography: {
    allVariants: {
      fontFamily: "Open Sans",
    },
  },
});

// markup
const IndexPage = () => {
  return (
    <ThemeProvider theme={theme}>
      <div className="root">
        <Header></Header>
        <div className={"main"}>
          <TitleCard></TitleCard>
          <GetStartedCard></GetStartedCard>
        </div>
      </div>
    </ThemeProvider>
  );
};

export default IndexPage;

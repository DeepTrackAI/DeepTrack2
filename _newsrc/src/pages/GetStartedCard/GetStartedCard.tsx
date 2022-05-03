import * as React from "react";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import { Button, Typography } from "@mui/material";

// Card component that displays a title and subtitle

const tabs = [
  {
    title: "Install",
    text: "Install DeepTrack from the command line.",
    action: "Placeholder",
  },
  {
    title: "Learn",
    text: "Learn more about DeepTrack through well documented tutorials. Play with the code on your own by running the code directly in your browser.",
    action: <Button variant="outlined">Open tutorials</Button>,
  },
  {
    title: "Documentation",
    text: "Read the documentation to learn more detailed information about DeepTrack.",
    action: <Button variant="outlined">Open documentation</Button>,
  },
  {
    title: "Source",
    text: "View the source code for DeepTrack.",
    action: <Button variant="outlined">Go to GitHub</Button>,
  },
];

const Label = (text: string) => (
  <Typography fontWeight={600} color="secondary">
    {text}
  </Typography>
);

const TitleCard = () => {
  // Card with a title and three tabs
  const [value, setValue] = React.useState(0);

  const handleChange = (
    _event: any,
    newValue: React.SetStateAction<number>
  ) => {
    setValue(newValue);
  };

  return (
    <div className="get-started-card">
      <p className="get-started-title">Get started with DeepTrack 2</p>
      <Tabs
        value={value}
        onChange={handleChange}
        aria-label="basic tabs example"
        className="get-started-tabs"
      >
        {tabs.map((tab, index) => (
          <Tab key={index} label={Label(tab.title)} />
        ))}
      </Tabs>

      <Typography className="get-started-body">{tabs[value].text}</Typography>

      <div className="get-started-action">{tabs[value].action}</div>
    </div>
  );
};

export default TitleCard;

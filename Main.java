package com.company;
import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws Exception {

        String n, h, d, dn, sn;
        String[] sd = {"cheated", "sad", "bored", "fearful", "embarrassed", "cheated", "hated", "belittled", "alone", "belittled", "demoralized", "derailed", "powerless", "singled out"};
        String[] dc={"Motivation.txt","Motivate1.txt"};
        int index;
        boolean retval,em;
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        System.out.println("HI How are you");
        h = br.readLine();

        System.out.println("How was your Day");
        d = br.readLine();

        System.out.println("What are you doing?");
        dn = br.readLine();

        System.out.println("You mind telling me Something about you present situation?");
        sn = br.readLine();

        System.out.println("How often do yo meet yup with friends");
        n= br.readLine();


        try {
            FileWriter myWriter = new FileWriter("D:\\DePaul University\\3rd Quarter\\Artificial Intelligence\\FinalProject\\UserResponse.txt");
            myWriter.write(h + "\n");
            myWriter.write(d + "\n");
            myWriter.write(dn + "\n");
            myWriter.write(sn + "\n");
            myWriter.write(n);
            myWriter.close();
            System.out.println("Successfully wrote to the file.");
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        String filePath = "D:\\DePaul University\\3rd Quarter\\Artificial Intelligence\\FinalProject\\Emotion.py";
        ProcessBuilder pb = new ProcessBuilder()           //Change the variables
                .command("python", "-u",filePath);
        Process p = pb.start();
        BufferedReader in = new BufferedReader(
                new InputStreamReader(p.getInputStream()));
        StringBuilder buffer = new StringBuilder();
        String line = null;
        while ((line = in.readLine()) != null) {
            buffer.append(line);
        }
        //int exitCode = p.waitFor();
        String ot = buffer.toString();
        em = Arrays.asList(sd).contains(ot);
        if(em)
           System.out.println("Emotion:"+"Sad");
        else
            System.out.println("Out of Scope");
        //System.out.println("Process exit value:" + exitCode);
        in.close();


        retval = Arrays.asList(sd).contains(ot);
        index = new Random().nextInt(dc.length);

        if (retval) {
            System.out.println("Don't feel sad.Life is all about ups and downs.Let me tell you a story.Type[y]");
            String ans = br.readLine();
            if (ans.equals("y")) {
                try {
                    br = new BufferedReader(new FileReader("D:\\DePaul University\\3rd Quarter\\Artificial Intelligence\\FinalProject\\"+dc[index]));


                    String contentLine = br.readLine();   //Change the code
                    while (contentLine != null) {
                        System.out.println(contentLine);
                        contentLine = br.readLine();
                    }
                }

                catch (IOException e) {
                    System.out.println("An error occurred.");
                    e.printStackTrace();
                }
            }

        }
        else{

            System.out.println("Nothing happened");
        }
    }
}




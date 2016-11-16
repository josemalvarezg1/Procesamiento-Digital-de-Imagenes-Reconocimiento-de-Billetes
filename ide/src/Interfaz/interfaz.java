/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Interfaz;

import Reconocimiento.reconocimiento;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingConstants;
import javax.swing.filechooser.FileNameExtensionFilter;
import org.opencv.core.Core;
import static Reconocimiento.reconocimiento.aplicarSURF;
import javax.swing.Icon;
import javax.swing.JOptionPane;
/**
 *
 * @authors     José Castro - CI 24635097
*               José Alvarez - CI 25038805
 */

public class interfaz extends javax.swing.JFrame {

    JFileChooser browsedFile = new JFileChooser();
    Image bitmapImage = null; 
    BufferedImage image = null;
    double bolivares = 0;
    
    public interfaz() {
        initComponents();
        browsedFile.addChoosableFileFilter(new FileNameExtensionFilter("Imágenes", "jpg", "png", "gif", "bmp"));
        jLabel1.setHorizontalAlignment(JLabel.CENTER);
        jLabel1.setVerticalAlignment(JLabel.CENTER);
        jTextField1.setEditable(false);
        botonProcesar.setEnabled(false);
        this.setExtendedState(JFrame.MAXIMIZED_BOTH); 
        //this.setUndecorated(true);
        this.setVisible(true);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jTextField1 = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        botonProcesar = new javax.swing.JButton();
        botonCargar = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        jLabel1 = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Reconocimiento de Billetes - José Alvarez, José Castro");

        jLabel2.setText("Bolívares");

        botonProcesar.setText("Procesar");
        botonProcesar.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                botonProcesarActionPerformed(evt);
            }
        });

        botonCargar.setText("Cargar");
        botonCargar.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                botonCargarActionPerformed(evt);
            }
        });

        jScrollPane1.setViewportView(jLabel1);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane1)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(botonCargar, javax.swing.GroupLayout.PREFERRED_SIZE, 75, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(18, 18, 18)
                        .addComponent(botonProcesar)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 485, Short.MAX_VALUE)
                        .addComponent(jLabel2)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(jTextField1, javax.swing.GroupLayout.PREFERRED_SIZE, 74, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 566, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jTextField1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel2)
                    .addComponent(botonProcesar)
                    .addComponent(botonCargar))
                .addGap(0, 0, 0))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void botonCargarActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_botonCargarActionPerformed
        browsedFile.showOpenDialog(this);
        File file = browsedFile.getSelectedFile();    
        if (file != null) {
            String pathImageFile = file.getAbsolutePath();  
            try {           
                image = ImageIO.read(file);
            } catch (IOException ex) { 
                Logger.getLogger(interfaz.class.getName()).log(Level.SEVERE, null, ex);
            }
            Image scaledImage = image.getScaledInstance(image.getWidth(),jLabel1.getHeight(),Image.SCALE_SMOOTH);
            jLabel1.setIcon(new ImageIcon(scaledImage));
            if (image == null) {
                botonProcesar.setEnabled(true);
            } else {   
                //En caso contrario, se muestra la imagen y se activan los botones   
                botonProcesar.setEnabled(true);
                bolivares = 0;
            }   
        }
    }//GEN-LAST:event_botonCargarActionPerformed

    private void botonProcesarActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_botonProcesarActionPerformed
        File file = browsedFile.getSelectedFile();  
        reconocimiento.archivo = null;
        String pathImageFile = file.getAbsolutePath();
        System.out.println("pathImageFile = " + pathImageFile);
        //Billetes Bs por delante
        while ( aplicarSURF("100_bolivares.jpg", pathImageFile) ) {
            bolivares += 100;
        }
        while ( aplicarSURF("50_bolivares.jpg", pathImageFile) ) {
            bolivares += 50;
        }
        while ( aplicarSURF("20_bolivares.jpg", pathImageFile) ) {
            bolivares += 20;
        }
        while ( aplicarSURF("10_bolivares.jpg", pathImageFile) ) {
            bolivares += 10;
        }
        while ( aplicarSURF("5_bolivares.jpg", pathImageFile) ) {
            bolivares += 5;
        }
        while ( aplicarSURF("2_bolivares.jpg", pathImageFile) ) {
            bolivares += 2;
        }
        
//        Billetes Bs por detrás
        while ( aplicarSURF("100_bolivares_reverso.jpg", pathImageFile) ) {
            bolivares += 100;
        }
        while ( aplicarSURF("50_bolivares_reverso.jpg", pathImageFile) ) {
            bolivares += 50;
        }
        while ( aplicarSURF("20_bolivares_reverso.jpg", pathImageFile) ) {
            bolivares += 20;
        }
        while ( aplicarSURF("10_bolivares_reverso.jpg", pathImageFile) ) {
            bolivares += 10;
        }
        while ( aplicarSURF("5_bolivares_reverso.jpg", pathImageFile) ) {
            bolivares += 5;
        }
        if ( aplicarSURF("2_bolivares_reverso.jpg", pathImageFile) ) {
            bolivares += 2;
        }
       // f.setVisible(false);
        jTextField1.setText( Double.toString(bolivares) );        
        
        String info = "Hay " + bolivares + " Bs.";
        System.out.println(info);        
        JOptionPane.showMessageDialog(new JFrame(), info, "Reconocimiento de billetes",JOptionPane.INFORMATION_MESSAGE);

    }//GEN-LAST:event_botonProcesarActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(interfaz.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(interfaz.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(interfaz.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(interfaz.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        System.loadLibrary("opencv_java2413");
        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new interfaz().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton botonCargar;
    private javax.swing.JButton botonProcesar;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JTextField jTextField1;
    // End of variables declaration//GEN-END:variables
}
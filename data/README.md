# **The Dataset**

*More information about the dataset is available [here](https://github.com/shahroudy/NTURGB-D)*.

___
___

### **Structure**

The **NTU RGB+D skeletons** contains 56,880 skeletal data samples (3D locations of 25 major body joints at each frame).

Each file in the dataset is in the format of SsssCcccPpppRrrrAaaa (e.g., S001C002P003R002A013), in which sss is the setup number (between 0001 and 017), ccc is the camera ID, ppp is the performer (or subject) ID, rrr is the replication number (1 or 2), and aaa is the action class label.

For more details about the setups, camera IDs, ... , please refer to the [NTU RGB+D dataset paper](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf).

___

### **Samples with missing skeletons**

302 samples have missing or incomplete skeleton data.
If you are working on skeleton-based analysis, please ignore these files in your training and testing procedures. The list of these are provided [here](https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt).  

___

### Sample codes

We have provided some MATLAB codes [here](https://github.com/shahroudy/NTURGB-D/tree/master/Matlab) to demonstrate how to read the skeleton files, map them to other modalities (RGB, depth, and IR frames), and visualize the skeleton data. The [codes](https://github.com/shahroudy/NTURGB-D/tree/master/Matlab) are suitable for both "NTU RGB+D" and "NTU RGB+D 120".

___

### **Action Classes**

The dataset contains 60 action classes listed below:

* A1. drink water. 
* A2. eat meal/snack. 
* A3. brushing teeth. 
* A4. brushing hair. 
* A5. drop. 
* A6. pickup. 
* A7. throw. 
* A8. sitting down. 
* A9. standing up (from sitting position). 
* A10. clapping. 
* A11. reading. 
* A12. writing. 
* A13. tear up paper. 
* A14. wear jacket. 
* A15. take off jacket. 
* A16. wear a shoe. 
* A17. take off a shoe. 
* A18. wear on glasses. 
* A19. take off glasses. 
* A20. put on a hat/cap. 
* A21. take off a hat/cap. 
* A22. cheer up. 
* A23. hand waving. 
* A24. kicking something. 
* A25. reach into pocket. 
* A26. hopping (one foot jumping). 
* A27. jump up. 
* A28. make a phone call/answer phone. 
* A29. playing with phone/tablet. 
* A30. typing on a keyboard. 
* A31. pointing to something with finger. 
* A32. taking a selfie. 
* A33. check time (from watch). 
* A34. rub two hands together. 
* A35. nod head/bow. 
* A36. shake head. 
* A37. wipe face. 
* A38. salute. 
* A39. put the palms together. 
* A40. cross hands in front (say stop). 
* A41. sneeze/cough. 
* A42. staggering. 
* A43. falling. 
* A44. touch head (headache). 
* A45. touch chest (stomachache/heart pain). 
* A46. touch back (backache). 
* A47. touch neck (neckache). 
* A48. nausea or vomiting condition. 
* A49. use a fan (with hand or paper)/feeling warm.
* A50. punching/slapping other person. 
* A51. kicking other person. 
* A52. pushing other person. 
* A53. pat on back of other person. 
* A54. point finger at the other person. 
* A55. hugging other person. 
* A56. giving something to other person. 
* A57. touch other person's pocket. 
* A58. handshaking. 
* A59. walking towards each other. 
* A60. walking apart from each other.
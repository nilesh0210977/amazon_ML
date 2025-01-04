-- phpMyAdmin SQL Dump
-- version 4.7.4
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 13, 2019 at 03:03 PM
-- Server version: 10.1.29-MariaDB
-- PHP Version: 7.2.0

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `qratt`
--

-- --------------------------------------------------------

--
-- Table structure for table `admininfo`
--

CREATE TABLE `admininfo` (
  `id` int(2) NOT NULL,
  `username` varchar(50) DEFAULT NULL,
  `password` varchar(200) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admininfo`
--

INSERT INTO `admininfo` (`id`, `username`, `password`) VALUES
(1, 'admin', '$2y$10$CUAhT8S7EmpJVVt7kN.O7OxKEHCA7S7HdraMy/6/wiZV6Y8LM0XXi');

-- --------------------------------------------------------

--
-- Table structure for table `classinfo`
--

CREATE TABLE `classinfo` (
  `classid` int(100) NOT NULL,
  `courseid` varchar(50) NOT NULL,
  `year` varchar(10) NOT NULL,
  `division` char(3) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `classinfo`
--

INSERT INTO `classinfo` (`classid`, `courseid`, `year`, `division`) VALUES
(1, '1', 'fy', 'a'),
(2, '2', 'fy', 'a');

-- --------------------------------------------------------

--
-- Table structure for table `courseinfo`
--

CREATE TABLE `courseinfo` (
  `courseid` tinyint(4) NOT NULL,
  `coursename` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `courseinfo`
--

INSERT INTO `courseinfo` (`courseid`, `coursename`) VALUES
(4, 'mech'),
(3, 'ece'),
(2, 'aids'),
(1, 'cse');

-- --------------------------------------------------------






-- ------------------------------------------------------
--
-- Table structure for table `cse`
--

CREATE TABLE `fycse-a` (
  `attid` int(5) NOT NULL,
  `date` date DEFAULT NULL,
  `time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `subid` int(3) DEFAULT NULL,
  `status` tinyint(2) NOT NULL DEFAULT '0',
  `r1` tinyint(1) NOT NULL DEFAULT '0',
  `c1` text,
  `r2` tinyint(1) NOT NULL DEFAULT '0',
  `c2` text,
  `r3` tinyint(1) NOT NULL DEFAULT '0',
  `c3` text
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `aids`
--

CREATE TABLE `fyaids-a` (
  `attid` int(5) NOT NULL,
  `date` date DEFAULT NULL,
  `time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `subid` int(3) DEFAULT NULL,
  `status` tinyint(2) NOT NULL DEFAULT '0',
  `r1` tinyint(1) NOT NULL DEFAULT '0',
  `c1` text,
  `r2` tinyint(1) NOT NULL DEFAULT '0',
  `c2` text,
  `r3` tinyint(1) NOT NULL DEFAULT '0',
  `c3` text
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `logincontrol`
--

CREATE TABLE `logincontrol` (
  `id` int(11) NOT NULL,
  `unixtime` int(15) NOT NULL,
  `username` text NOT NULL,
  `ip` text NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `studentinfo`
--

CREATE TABLE `studentinfo` (
  `id` int(11) NOT NULL,
  `rollno` smallint(5) NOT NULL,
  `name` tinytext NOT NULL,
  `username` varchar(200) DEFAULT NULL,
  `password` varchar(200) NOT NULL,
  `email` text NOT NULL,
  `classid` smallint(6) NOT NULL,
  `status` tinyint(4) NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `studentinfo`
--

INSERT INTO `studentinfo` (`id`, `rollno`, `name`, `username`, `password`, `email`, `classid`, `status`) VALUES
(20, 1, 'Nilesh Sharma', 'nilesh_sharma', '$2a$12$mKqvTlDJU5MMD1.jmQPe0OtecSPHaOGMJtq0.WLl39Nx5iys7p9xm', '', 1, 0),
(02, 2, 'Ashu Rajput', 'ashu_rajput', '$2a$12$.UcWSRHWrQSonTMIwmOYNu3BmLJU6PQbICjmIMRJ9rBKhNxSBW2ky', '', 2, 0);

-- --------------------------------------------------------

--
-- Table structure for table `subjectinfo`
--

CREATE TABLE `subjectinfo` (
  `subjectid` int(100) NOT NULL,
  `subname` varchar(100) NOT NULL,
  `classid` int(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `subjectinfo`
--

INSERT INTO `subjectinfo` (`subjectid`, `subname`, `classid`) VALUES
(1, 'Object Oriented Programing', 1),
(2, 'DSA', 1),
(3, 'DAA', 1),
(4, 'DBMS', 2),
(5, 'DCCN', 2),
(6, 'TOC', 2);

-- --------------------------------------------------------

--
-- Table structure for table `sybscit-b`
--

CREATE TABLE `sycse-a` (
  `attid` int(5) NOT NULL,
  `date` date DEFAULT NULL,
  `time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `subid` int(3) DEFAULT NULL,
  `status` tinyint(2) NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `teacherinfo`
--

CREATE TABLE `teacherinfo` (
  `tid` int(50) NOT NULL,
  `name` varchar(50) NOT NULL,
  `username` varchar(50) NOT NULL,
  `password` varchar(100) NOT NULL,
  `rights` char(2) NOT NULL DEFAULT 'T',
  `sub1` int(100) DEFAULT NULL,
  `sub2` int(100) DEFAULT NULL,
  `sub3` int(100) DEFAULT NULL,
  `sub4` int(100) DEFAULT NULL,
  `sub5` int(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `teacherinfo`
--

INSERT INTO `teacherinfo` (`tid`, `name`, `username`, `password`, `rights`, `sub1`, `sub2`, `sub3`, `sub4`, `sub5`) VALUES
(1, 'teacher1', 'teacher_1', '$2y$10$9c8yODI3SLe.xRcFzvyrVede3bfUCQBYrxrrGCeJKg/qvm2BQUXKi', 'a', 1, 2, 3, NULL, NULL),
(2, 'teacher2', 'teacher_2', '$2y$10$9c8yODI3SLe.xRcFzvyrVede3bfUCQBYrxrrGCeJKg/qvm2BQUXKi', 'a', 4, 5, 4, NULL, NULL);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `admininfo`
--
ALTER TABLE `admininfo`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `classinfo`
--
ALTER TABLE `classinfo`
  ADD PRIMARY KEY (`classid`),
  ADD UNIQUE KEY `courseid` (`courseid`,`year`,`division`);

--
-- Indexes for table `courseinfo`
--
ALTER TABLE `courseinfo`
  ADD PRIMARY KEY (`courseid`),
  ADD UNIQUE KEY `coursename` (`coursename`);

--
-- Indexes for table `fybcom-a`
--
ALTER TABLE `fyaids-a`
  ADD PRIMARY KEY (`attid`);

--
-- Indexes for table `fybscit-a`
--
ALTER TABLE `fycse-a`
  ADD PRIMARY KEY (`attid`);

--
-- Indexes for table `logincontrol`
--
ALTER TABLE `logincontrol`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `id` (`id`);

--
-- Indexes for table `studentinfo`
--
ALTER TABLE `studentinfo`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `id` (`id`),
  ADD UNIQUE KEY `username` (`username`(64));

--
-- Indexes for table `subjectinfo`
--
ALTER TABLE `subjectinfo`
  ADD PRIMARY KEY (`subjectid`);

--
-- Indexes for table `sybscit-b`
--
ALTER TABLE `sycse-a`
  ADD PRIMARY KEY (`attid`);

--
-- Indexes for table `teacherinfo`
--
ALTER TABLE `teacherinfo`
  ADD PRIMARY KEY (`tid`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `admininfo`
--
ALTER TABLE `admininfo`
  MODIFY `id` int(2) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `classinfo`
--
ALTER TABLE `classinfo`
  MODIFY `classid` int(100) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `courseinfo`
--
ALTER TABLE `courseinfo`
  MODIFY `courseid` tinyint(4) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `fybcom-a`
--
ALTER TABLE `fyaids-a`
  MODIFY `attid` int(5) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `fybscit-a`
--
ALTER TABLE `fycse-a`
  MODIFY `attid` int(5) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `logincontrol`
--
ALTER TABLE `logincontrol`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `subjectinfo`
--
ALTER TABLE `subjectinfo`
  MODIFY `subjectid` int(100) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;

--
-- AUTO_INCREMENT for table `sybscit-b`
--
ALTER TABLE `sycse-a`
  MODIFY `attid` int(5) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `teacherinfo`
--
ALTER TABLE `teacherinfo`
  MODIFY `tid` int(50) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;

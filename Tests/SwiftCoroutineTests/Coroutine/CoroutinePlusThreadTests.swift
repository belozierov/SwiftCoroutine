////
////  CoroutinePlusThreadTests.swift
////  SwiftCoroutine
////
////  Created by Alex Belozierov on 21.01.2020.
////  Copyright © 2020 Alex Belozierov. All rights reserved.
////
//
//import XCTest
//@testable import SwiftCoroutine
//
//class CoroutinePlusThreadTests: XCTestCase {
//    
//    func testCurrentCoroutine() {
//        let coroutine1 = Coroutine()
//        let coroutine2 = Coroutine()
//        XCTAssertNil(try? Coroutine.current())
//        XCTAssertFalse(Coroutine.isInsideCoroutine)
//        XCTAssertFalse(coroutine1.isCurrent)
//        XCTAssertFalse(coroutine2.isCurrent)
//        coroutine1.start {
//            XCTAssertTrue(coroutine1.isCurrent)
//            XCTAssertFalse(coroutine2.isCurrent)
//            XCTAssertEqual(try? Coroutine.current(), coroutine1)
//            XCTAssertTrue(Coroutine.isInsideCoroutine)
//            coroutine2.start {
//                XCTAssertFalse(coroutine1.isCurrent)
//                XCTAssertTrue(coroutine2.isCurrent)
//                XCTAssertEqual(try? Coroutine.current(), coroutine2)
//                XCTAssertTrue(Coroutine.isInsideCoroutine)
//            }
//            XCTAssertTrue(coroutine1.isCurrent)
//            XCTAssertFalse(coroutine2.isCurrent)
//            XCTAssertEqual(try? Coroutine.current(), coroutine1)
//            XCTAssertTrue(Coroutine.isInsideCoroutine)
//        }
//        XCTAssertNil(try? Coroutine.current())
//        XCTAssertFalse(Coroutine.isInsideCoroutine)
//        XCTAssertFalse(coroutine1.isCurrent)
//        XCTAssertFalse(coroutine2.isCurrent)
//    }
//    
//    func testPerformAsCurrent() {
//        let coroutine = Coroutine()
//        XCTAssertFalse(coroutine.isCurrent)
//        XCTAssertEqual(Thread.current.currentCoroutine, nil)
//        coroutine.performAsCurrent {
//            XCTAssertTrue(coroutine.isCurrent)
//            XCTAssertEqual(Thread.current.currentCoroutine, coroutine)
//            XCTAssertEqual(try? Coroutine.current(), coroutine)
//        }
//        XCTAssertFalse(coroutine.isCurrent)
//        XCTAssertEqual(Thread.current.currentCoroutine, nil)
//    }
//    
//}

////
////  CorotuneDelayTests.swift
////  SwiftCoroutine
////
////  Created by Alex Belozierov on 21.01.2020.
////  Copyright © 2020 Alex Belozierov. All rights reserved.
////
//
//import XCTest
//@testable import SwiftCoroutine
//
//class CorotuneDelayTests: XCTestCase {
//    
//    func testDelay() {
//        let exp = expectation(description: "testDelay")
//        let date = Date()
//        let coroutine = Coroutine()
//        coroutine.start {
//            try? Coroutine.delay(.now() + 1)
//            try? Coroutine.delay(1)
//            XCTAssertDuration(from: date, in: 2..<3)
//            exp.fulfill()
//        }
//        wait(for: [exp], timeout: 3)
//    }
//    
//}

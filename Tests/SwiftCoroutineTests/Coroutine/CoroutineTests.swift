//
//  CoroutineTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoroutineTests: XCTestCase {
    
    func testAwait() {
        let exp = expectation(description: "testAwait")
        exp.expectedFulfillmentCount = 1000
        let dispatcher = CoroutineDispatcher(scheduler: .immediate)
        for i in 0..<1000 {
            dispatcher.execute {
                var a = Int.random(in: 0..<1000)
                let b: Int = try! TaskScheduler.global.await { completion in
                    if i == 0 { sleep(1) }
                    a += Int.random(in: 0..<1000)
                    DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                        a += Int.random(in: 0..<1000)
                        completion(Int.random(in: 0..<1000))
                    }
                } + a
                exp.fulfill()
            }
        }
        wait(for: [exp], timeout: 10)
    }

//    func testSyncContext() {
//        let exp = XCTOrderedExpectation(count: 5)
//        let coroutine1 = Coroutine {
//            exp.fulfill(0)
//            try? Coroutine.suspend()
//            exp.fulfill(3)
//        }
//        try? coroutine1.resume()
//        exp.fulfill(1)
//        try? Coroutine {
//            exp.fulfill(2)
//            try? coroutine1.resume()
//            exp.fulfill(4)
//        }.resume()
//        wait(for: exp, timeout: 1)
//    }
    
//    func testRestartWithDispatcher() {
//        let exp = expectation(description: "testRestartWithDispatcher")
//        let coroutine = Coroutine()
//        coroutine.start {
//            XCTAssertTrue(Thread.isMainThread)
//            coroutine.restart(with: .global)
//            XCTAssertFalse(Thread.isMainThread)
//            exp.fulfill()
//        }
//        wait(for: [exp], timeout: 1)
//    }
//
    
//    func testCurrent() {
//        XCTAssertNil(try? Coroutine.current())
//        XCTAssertFalse(Coroutine.isInsideCoroutine)
//        try? Coroutine {
//            XCTAssertNotNil(try? Coroutine.current())
//            XCTAssertTrue(Coroutine.isInsideCoroutine)
//            XCTAssertTrue(try? Coroutine.current().isCurrent)
//        }.resume()
//    }
    
//    func testStates() {
//        let exp = expectation(description: "testStates")
//        let coroutine = Coroutine {
//            let coroutine = try? Coroutine.current()
//            XCTAssertEqual(coroutine?.state, .running)
//            try? Coroutine.suspend {
//                XCTAssertEqual(coroutine?.state, .suspended)
//                exp.fulfill()
//            }
//            XCTAssertEqual(coroutine?.state, .running)
//        }
//        XCTAssertEqual(coroutine.state, .prepared)
//        try? coroutine.resume()
//        XCTAssertEqual(coroutine.state, .suspended)
//        try? coroutine.resume()
//        XCTAssertEqual(coroutine.state, .finished)
//        wait(for: [exp], timeout: 1)
//    }
    
    func testStackSize() {
        XCTAssertEqual(Coroutine.StackSize(size: 234).size, 234)
        XCTAssertEqual(Coroutine.StackSize.pages(4).size, 4 * .pageSize)
    }

}

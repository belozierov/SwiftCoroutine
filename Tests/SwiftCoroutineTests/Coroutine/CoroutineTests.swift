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
                a += try! TaskScheduler.global.await { completion in
                    if i == 0 { sleep(1) }
                    a += Int.random(in: 0..<1000)
                    DispatchQueue.global().asyncAfter(deadline: .now() + 2) {
                        a += Int.random(in: 0..<1000)
                        completion(Int.random(in: 0..<1000))
                    }
                }
                exp.fulfill()
            }
        }
        wait(for: [exp], timeout: 10)
    }
    
    func testStackSize() {
        XCTAssertEqual(Coroutine.StackSize(size: 234).size, 234)
        XCTAssertEqual(Coroutine.StackSize.pages(4).size, 4 * .pageSize)
    }
    
    func testDelay() {
        let exp = expectation(description: "testDelay")
        let date = Date()
        CoroutineDispatcher.global.execute {
            try? Coroutine.delay(.now() + 1)
            XCTAssertDuration(from: date, in: 1..<2)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 3)
    }

}

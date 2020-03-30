//
//  TaskSchedulerTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 28.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class TaskSchedulerTests: XCTestCase {
    
    func testAbc() {
        measure {
            var counter = 0
            for _ in 0..<100_000 {
                DispatchQueue.main.scheduleTask {
                    counter += 1
                }
            }
            XCTAssertEqual(counter, 100_000)
        }
    }
    
}
